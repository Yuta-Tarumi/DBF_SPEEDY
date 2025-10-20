import numpy as np
import sys
import pickle
import torch
import tracemalloc
from numpy.random import MT19937, RandomState, SeedSequence
#from scipy.integrate import solve_ivp
from torch.utils.data import Dataset
import bisect, json, os, psutil
from einops import rearrange
from pathlib import Path
import blosc2
import gc
from memory_profiler import profile
import logging, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def load_blosc2(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return blosc2.unpack_array2(f.read())

class SPEEDY(Dataset):
    def __init__(
        self,
        index_json: str,
        n_steps: int,
        num_data: int,
        device: str = "cuda",
        fixed_noise: bool = False,
        obs_pattern: str = "Koopman"
    ):
        super().__init__()
        self.root = Path("/data_prediction001/ytarumi/data")
        self.n_steps = n_steps
        self.device = device
        self.fixed_noise = fixed_noise
        self.obs_pattern = obs_pattern
        self.num_data = num_data
        self.phi_mean = np.load(f"{self.root}/data_mean/phi_mean.npy").astype(np.float32)
        self.ps_mean = np.load(f"{self.root}/data_mean/ps_mean.npy").astype(np.float32)
        self.t_mean = np.load(f"{self.root}/data_mean/t_mean.npy").astype(np.float32)
        self.fixed_noise = fixed_noise

        with open(index_json) as f:
            self.dir2files = json.load(f)
        self.dirs = list(self.dir2files.keys())
        self.starts = [0]
        for d in self.dirs:
            self.starts.append(self.starts[-1] + len(self.dir2files[d]))
        # total number of samples
        self._len = self.starts[-1]

    def _load(self, p: Path) -> np.ndarray:
        """Read one Blosc2-compressed slice → ndarray (float32)."""
        with open(p, "rb") as fh:
            return blosc2.unpack_array2(fh.read())

    def __getitem__(self, index: int):
        # ── locate directory & inner index (same as before) ──────────────────
        d_idx  = bisect.bisect_right(self.starts, index) - 1
        inner  = index - self.starts[d_idx]
        dname  = self.dirs[d_idx]
        filestem = self.dir2files[dname][inner]      # "ps_data_index123456"
        dpath  = self.root / dname                   # /.../data_train17
        
        # ── grab just the numeric part once ──────────────────────────────────
        num = filestem.split("ps_data_index")[1]     # "123456"
        #print(f"{index=} {num=}")
        
        # ── build correct filenames for each variable ────────────────────────
        ps_path  = dpath / f"ps_data/ps_data_index{num}"
        q_path   = dpath / f"q_data/q_data_index{num}"
        t_path   = dpath / f"t_data/t_data_index{num}"
        u_path   = dpath / f"u_data/u_data_index{num}"
        v_path   = dpath / f"v_data/v_data_index{num}"
        #phi_path = dpath / f"phi_data/phi_data_index{num}"
        
        # ── load & normalise (variable names unchanged) ──────────────────────
        data_z_ps  = (self._load(ps_path ).reshape(self.n_steps, 4608 ) - self.ps_mean) / 1e4
        data_z_q   =  self._load(q_path  ).reshape(self.n_steps, 36864) / 0.003
        data_z_t   = (self._load(t_path  ).reshape(self.n_steps, 36864) - self.t_mean) / 20
        data_z_u   =  self._load(u_path  ).reshape(self.n_steps, 36864) / 15
        data_z_v   =  self._load(v_path  ).reshape(self.n_steps, 36864) / 15
        #data_z_phi = (self._load(phi_path).reshape(self.n_steps, 36864) - self.phi_mean) / 200
        
        # ─── reshape to (time, channel, latlon) tensors ───────────────────
        data_z_ps_out  = torch.tensor(rearrange(data_z_ps,  "t (ll)      -> t 1 ll"))
        data_z_q_out   = torch.tensor(rearrange(data_z_q,   "t (h ll)    -> t h ll", h=8))
        data_z_t_out   = torch.tensor(rearrange(data_z_t,   "t (h ll)    -> t h ll", h=8))
        data_z_u_out   = torch.tensor(rearrange(data_z_u,   "t (h ll)    -> t h ll", h=8))
        data_z_v_out   = torch.tensor(rearrange(data_z_v,   "t (h ll)    -> t h ll", h=8))
        #data_z_phi_out = torch.tensor(rearrange(data_z_phi, "t (h ll)    -> t h ll", h=8))

        data_z = torch.cat([data_z_ps_out, data_z_u_out, data_z_v_out, data_z_t_out, data_z_q_out], dim=1)
        if self.obs_pattern == "Koopman":
            data_o = rearrange(np.array([data_z_u, data_z_v, data_z_q, data_z_t]), "a b c -> b (a c)") # Koopman
        elif self.obs_pattern == "dense":
            data_o_u = rearrange((torch.tensor(rearrange(data_z_u*15, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, 2:46:2, ::2])+torch.randn(self.n_steps, 8, 22, 48)), "t h lat lon -> t (h lat lon)")/15
            data_o_v = rearrange((torch.tensor(rearrange(data_z_v*15, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, 2:46:2, ::2])+torch.randn(self.n_steps, 8, 22, 48)), "t h lat lon -> t (h lat lon)")/15
            data_o_t = rearrange((torch.tensor(rearrange(data_z_t*20, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, 2:46:2, ::2])+torch.randn(self.n_steps, 8, 22, 48)), "t h lat lon -> t (h lat lon)")/20
            data_o_q = rearrange((torch.tensor(rearrange(data_z_q*0.003, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, 2:46:2, ::2])+1e-4*torch.randn(self.n_steps, 8, 22, 48)), "t h lat lon -> t (h lat lon)")/0.003
            data_o = rearrange(np.array([data_o_u, data_o_v, data_o_t, data_o_q]), "a b c -> b (a c)") # Koopman
        elif self.obs_pattern == "sparse":
            data_o_ps = rearrange((torch.tensor(rearrange(data_z_ps*1e4, "t (lat lon) -> t lat lon", lat=48, lon=96)[:, 1:46:4, ::4])+100*torch.randn(self.n_steps, 12, 24)), "t lat lon -> t 1 (lat lon)")/1e4
            data_o_u = rearrange((torch.tensor(rearrange(data_z_u*15, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, 1:46:4, ::4])+torch.randn(self.n_steps, 8, 12, 24)), "t h lat lon -> t h (lat lon)")/15
            data_o_v = rearrange((torch.tensor(rearrange(data_z_v*15, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, 1:46:4, ::4])+torch.randn(self.n_steps, 8, 12, 24)), "t h lat lon -> t h (lat lon)")/15
            data_o_t = rearrange((torch.tensor(rearrange(data_z_t*20, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, 1:46:4, ::4])+torch.randn(self.n_steps, 8, 12, 24)), "t h lat lon -> t h (lat lon)")/20
            data_o_q = rearrange((torch.tensor(rearrange(data_z_q*0.003, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, 1:46:4, ::4])+1e-4*torch.randn(self.n_steps, 8, 12, 24)), "t h lat lon -> t h (lat lon)")/0.003
            data_o = torch.cat([data_o_ps, data_o_u, data_o_v, data_o_t, data_o_q[:, :4, :]], dim=1)
        elif self.obs_pattern == "sparser":
            data_o_ps = rearrange((torch.tensor(rearrange(data_z_ps*1e4, "t (lat lon) -> t lat lon", lat=48, lon=96)[:, 4:48:8, ::8])+100*torch.randn(self.n_steps, 6, 12)), "t lat lon -> t 1 (lat lon)")/1e4
            data_o_u = rearrange((torch.tensor(rearrange(data_z_u*15, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, 4:48:8, ::8])+torch.randn(self.n_steps, 8, 6, 12)), "t h lat lon -> t h (lat lon)")/15
            data_o_v = rearrange((torch.tensor(rearrange(data_z_v*15, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, 4:48:8, ::8])+torch.randn(self.n_steps, 8, 6, 12)), "t h lat lon -> t h (lat lon)")/15
            data_o_t = rearrange((torch.tensor(rearrange(data_z_t*20, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, 4:48:8, ::8])+torch.randn(self.n_steps, 8, 6, 12)), "t h lat lon -> t h (lat lon)")/20
            data_o_q = rearrange((torch.tensor(rearrange(data_z_q*0.003, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, 4:48:8, ::8])+1e-4*torch.randn(self.n_steps, 8, 6, 12)), "t h lat lon -> t h (lat lon)")/0.003
            data_o = torch.cat([data_o_ps, data_o_u, data_o_v, data_o_t, data_o_q[:, :4, :]], dim=1)

        elif self.obs_pattern == "sparsest":
            data_o_ps = rearrange((torch.tensor(rearrange(data_z_ps*1e4, "t (lat lon) -> t lat lon", lat=48, lon=96)[:, 7:48:16, ::16])+100*torch.randn(self.n_steps, 3, 6)), "t lat lon -> t 1 (lat lon)")/1e4
            data_o_u = rearrange((torch.tensor(rearrange(data_z_u*15, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, 7:48:16, ::16])+torch.randn(self.n_steps, 8, 3, 6)), "t h lat lon -> t h (lat lon)")/15
            data_o_v = rearrange((torch.tensor(rearrange(data_z_v*15, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, 7:48:16, ::16])+torch.randn(self.n_steps, 8, 3, 6)), "t h lat lon -> t h (lat lon)")/15
            data_o_t = rearrange((torch.tensor(rearrange(data_z_t*20, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, 7:48:16, ::16])+torch.randn(self.n_steps, 8, 3, 6)), "t h lat lon -> t h (lat lon)")/20
            data_o_q = rearrange((torch.tensor(rearrange(data_z_q*0.003, "t (h lat lon) -> t h lat lon", h=8, lat=48, lon=96)[:, :, 7:48:16, ::16])+1e-4*torch.randn(self.n_steps, 8, 3, 6)), "t h lat lon -> t h (lat lon)")/0.003
            data_o = torch.cat([data_o_ps, data_o_u, data_o_v, data_o_t, data_o_q[:, :4, :]], dim=1)

        return (data_o, data_z)

    def __len__(self) -> int:
        return self.num_data
