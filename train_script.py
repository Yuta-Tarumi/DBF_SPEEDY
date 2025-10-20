
import numpy as np
import torch

import pickle
import time
import torch.nn as nn
import torch.nn.functional as F
from DBF.model.dbf import DeepBayesianFilterBlockDiag
from DBF.dataset.speedy import SPEEDY
from torch.utils.data import default_collate, get_worker_info, DataLoader
from einops import rearrange
import torch.multiprocessing as mp
import tqdm
import random
import json, tempfile, os, psutil, pathlib
import shutil
from read_config import read_config
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import BatchSampler, Sampler, SequentialSampler
import argparse

import logging
import logging.handlers
import queue

mp.set_start_method('spawn', force=True)
torch.set_float32_matmul_precision("high")
#torch.cuda.set_device(7)

import smtplib
from email.mime.text import MIMEText
import tracemalloc

os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
print(f"{os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO']=}")

PURGE_INTERVAL = 60          # seconds   (10 min)
LOG_DIR = pathlib.Path("/home/ytarumi/DBF_work/memlogs")
LOG_DIR.mkdir(exist_ok=True)

_last_purge = {}              # worker_id ➜ last-purge timestamp
_log_fp     = {}              # worker_id ➜ open file handle

def _pinned_mib():
    # key name is the same in 2.6 / 2.7
    return torch.cuda.memory_stats().get("pinned.current", 0) / 2**20

def _fmt(ts: float) -> str:
    # Asia/Tokyo (JST) is the system zone on prediction001
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

def worker_init_fn(worker_id):
    tracemalloc.start(15)
    import gc
    #gc.disable()
    gc.enable()

    proc = psutil.Process(os.getpid())
    print(f"[worker {worker_id:02d}] PID {os.getpid()}  "
          f"initial RSS = {proc.memory_info().rss/2**20:.2f} MiB", flush=True)

    # deterministic seed per worker (optional)
    seed = torch.initial_seed() % 2**32
    torch.manual_seed(seed);  torch.cuda.manual_seed_all(seed)

    torch.cuda.empty_cache()              # initial purge
    now = time.time()
    _last_purge[worker_id] = now

    # ---------- open per-worker log ----------
    fn = LOG_DIR / f"worker{worker_id:02d}_{os.getpid()}.tsv"
    fp = fn.open("w", buffering=1)
    fp.write("t_s\trss_mib\tpinned_mib\n")
    rss = proc.memory_info().rss / 2**20
    fp.write(f"{_fmt(now)}\t{proc.memory_info().rss/2**20:.1f}\t {_pinned_mib():.1f}\n")
    _log_fp[worker_id] = fp
    # -----------------------------------------

def dump_top(prefix: str, limit: int = 30):
    snap  = tracemalloc.take_snapshot()
    stats = snap.statistics('lineno')[:limit]
    path  = pathlib.Path(f"{prefix}_{int(time.time())}.pkl")
    path.write_bytes(pickle.dumps(stats))
    print(f"[worker] wrote snapshot {path}")
    
def collate(batch):
    """
    Custom collate_fn for the DataLoader.
    Runs inside every worker process.
    """
    info = torch.utils.data.get_worker_info()        # → None in the main proc
    if info is not None:                             # we ARE in a worker
        wid   = info.id
        proc  = psutil.Process(os.getpid())
        now   = time.time()

        # ── 1.  Timed purge of pinned-memory cache + GC  ──────────────
        if now - _last_purge.get(wid, 0.0) >= PURGE_INTERVAL:
            torch.cuda.empty_cache()                 # free idle pinned blocks
            #import gc;  gc.collect()                 # reclaim Python objects
            _last_purge[wid] = now

            fp = _log_fp.get(wid)
            if fp is not None:
                rss = proc.memory_info().rss / 2**20
                fp.write(f"{_fmt(now)}\tPURGED\tRSS={rss:.1f}\n")

        # ── 2.  One log line for every batch  ─────────────────────────
        fp = _log_fp.get(wid)
        if fp is not None:
            rss    = proc.memory_info().rss / 2**20
            pinned = _pinned_mib()                  # may still be 0 here
            fp.write(f"{_fmt(now)}\t{rss:.1f}\t{pinned:.1f}\n")

        #dump_top(f"memlogs/trace_worker{wid:02d}")

    # Default collate (stacks or combines tensors)
    return default_collate(batch)

def send_epoch_email(loss_integral, loss_kl, testloss1_integral, testloss1_kl, testloss2_integral, testloss2_kl, log_R, log_Q, epoch, from_addr, to_addr, smtp_user, smtp_password):
    """
    Sends an email with the loss and epoch information.
    
    Parameters:
      loss (float): The loss value.
      epoch (int): The epoch number.
      from_addr (str): The sender's email address.
      to_addr (str): The recipient's email address.
      smtp_user (str): The SMTP username (usually the same as from_addr).
      smtp_password (str): The SMTP password or app-specific password.
    """
    subject = f"Epoch {epoch} Completed"
    body = f"Epoch: {epoch}\nTrain loss integral: {loss_integral}\nTrain loss KL: {loss_kl}\nTest loss integral1: {testloss1_integral}\nTest loss KL1: {testloss1_kl}\nTest loss integral2: {testloss2_integral}\nTest loss KL2: {testloss2_kl}\nlog_R: {log_R}\nlog_Q: {log_Q}"
    
    # Create a MIMEText object with the message body.
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_addr
    msg['To'] = to_addr

    # Connect to Gmail's SMTP server with TLS
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()  # Upgrade the connection to secure TLS
        server.login(smtp_user, smtp_password)
        server.sendmail(from_addr, [to_addr], msg.as_string())

def lr_lambda(current_step: int, warmup_steps: int, lr_scaling_steps: int):
    if current_step < warmup_steps:
        return float(current_step)/float(max(1, warmup_steps))
    
    return pow(10, -(current_step-warmup_steps)/(lr_scaling_steps-warmup_steps))

def setup_logger(name, log_file, level=logging.INFO):
    """
    Set up an asynchronous logger:
      - A QueueHandler immediately enqueues log records.
      - A QueueListener running in a background thread dequeues records and writes them via a file handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    log_queue = queue.Queue()  # unbounded queue
    q_handler = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(q_handler)
    
    # FileHandler to write log records to the specified file.
    f_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(message)s")
    f_handler.setFormatter(formatter)
    
    # QueueListener listens on log_queue and sends records to f_handler.
    listener = logging.handlers.QueueListener(log_queue, f_handler)
    listener.start()
    return logger, listener

def save_obs_noise(param: torch.Tensor,
                   name: str,
                   step: int | str,
                   out_dir: str = "obs_noise_snapshots") -> str:
    """
    Save a per-variable log-variance tensor to <out_dir>/log_R_<name>_<step>.npy
    and return the file path.

    Parameters
    ----------
    param : torch.Tensor
        The `nn.Parameter` you want to dump (e.g. model.log_R_u).
    name : str
        Variable tag: "ps", "u", "v", "t", "q", or "phi".
    step : int or str
        Identifier appended to the file name (epoch, global_step, …).
    out_dir : str, optional
        Folder for the snapshots. Created automatically.

    Returns
    -------
    str
        Absolute path of the written `.npy` file.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"log_R_{name}_{int(step):06d}.npy")
    np.save(path, param.detach().cpu().numpy())
    return path

class DirBlockSampler(Sampler):
    def __init__(self, index_json, shuffle_within=True, max_samples=None):
        self.map = json.load(open(index_json))
        self.max_samples = max_samples
        self.dirs   = list(self.map.keys())                 # outer order fixed
        self.starts = [0]                                   # cumulative offsets
        s = 0
        for d in self.dirs:
            s += len(self.map[d])
            self.starts.append(s)
        self.shuffle_within = shuffle_within
        if self.max_samples is None:
            self._len = self.starts[-1]
        else:
            self._len = self.max_samples


    def __len__(self):
        return self._len
            
    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        rng = random.Random(worker.seed if worker else None)

        yielded = 0
        for d, start in zip(self.dirs, self.starts):        # directory loop
            order = list(range(len(self.map[d])))
            if self.shuffle_within:
                rng.shuffle(order)                          # random *inside* dir
            for off in order:
                if self.max_samples is not None and yielded >= self.max_samples:
                    return
                yield start + off                           # global index
                yielded += 1

def main():
    parser = argparse.ArgumentParser(
        description="Deep Bayesian Filter for SPEEDY model"
    )                                                                                                                                          
    parser.add_argument("--config", type=str, help="config file path", required=True)
    parser.add_argument("--niter", type=int, help="number of current iterations", required=True)
    args = parser.parse_args()
    config_file = args.config
    niter = args.niter
    config = read_config(f"{config_file}")
    
    latent_dim = int(config["model"]["latent_dim"])
    T = int(config["model"]["T"])
    batch_size = int(config["model"]["batch_size"])
    num_epoch = int(config["model"]["num_epoch"])
    lr = float(config["model"]["lr"])
    dropout = float(config["model"]["dropout"])
    max_norm = float(config["model"]["max_norm"])
    arch = config["model"]["arch"]
    warmup_steps = int(config["model"]["warmup_steps"])
    lr_scaling_steps = int(config["model"]["lr_scaling_steps"])
    model_seed = int(config["model"]["model_seed"])
    outdir = config["model"]["outdir"]
    q_distribution = config["model"]["q_distribution"]
    obs_setting = config["data"]["obs_setting"]
    gpu_index = int(config["others"]["gpu_index"])
    torch.cuda.set_device(gpu_index)
    simulation_dimensions = [8, 48, 96]
    
    N_train = 4000*365
    finaliter = int(N_train//batch_size)

    trainset = SPEEDY(num_data=N_train, index_json="index_train.json", n_steps=T, obs_pattern=obs_setting, device="cuda")
    testset1 = SPEEDY(num_data=100, index_json="index_test1.json", n_steps=T, obs_pattern=obs_setting, device="cuda",
                      fixed_noise=True)
    testset2 = SPEEDY(num_data=32, index_json="index_test2.json", n_steps=T, obs_pattern=obs_setting, device="cuda",
                      fixed_noise=True)

    train_sampler = DirBlockSampler("index_train.json", shuffle_within=True, max_samples=N_train)
    batch_sampler   = BatchSampler(train_sampler,
                                   batch_size=batch_size,
                                   drop_last=False)
    test_sampler1 = SequentialSampler(testset1)
    test_sampler2 = SequentialSampler(testset2)
    dataloader = DataLoader(
        trainset,
        batch_sampler=batch_sampler,
        num_workers=16, # reduce if data fetching is not slow
        persistent_workers=True,
        prefetch_factor=4, # reduce if data fetching is not slow
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate
    )
    print(f"{len(dataloader)=}")
    test_batch_size = 20
    testloader1 = DataLoader(
        testset1,
        batch_size=test_batch_size,
        sampler=SequentialSampler(testset1),
        shuffle=False,                      
        num_workers=4,
        persistent_workers=True,            
        pin_memory=True,                    
        prefetch_factor=2,                  
        worker_init_fn=worker_init_fn,
    )
    testloader2 = DataLoader(
        testset2,
        batch_size=test_batch_size,
        sampler=SequentialSampler(testset2),
        shuffle=False,              
        num_workers=4,
        persistent_workers=True,    
        pin_memory=True,            
        prefetch_factor=2,          
        worker_init_fn=worker_init_fn,
    )
    print(f"{len(train_sampler)=}") 
    print(f"{len(batch_sampler)=}") 
    print(f"{len(dataloader)=}")  
    
    print("dataloader ok") 
    model = DeepBayesianFilterBlockDiag(
        latent_dim=latent_dim,
        simulation_dimensions=simulation_dimensions,
        model_seed=model_seed,
        arch=arch,
        dropout=dropout,
        q_distribution=q_distribution
    ).cuda()
    if niter > 0:
        print(f"{niter=}, loading model weights of niter={niter-1}")
        state_dict = torch.load(os.path.join(outdir, f"model_weights/model_iter{niter-1}_epoch0")) # load weights of the previous iteration
        model.load_state_dict(state_dict)
    # optimizers 
    optimizer = torch.optim.AdamW(model.parameters(), eps=1e-7, lr=lr)
    scaler = torch.amp.GradScaler("cuda")
    print(f"{optimizer=}")
    scheduler = LambdaLR(optimizer, partial(lr_lambda, warmup_steps=warmup_steps, lr_scaling_steps=lr_scaling_steps))
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "model_outputs"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "model_weights"), exist_ok=True)
    for filename in [f"logs/lr_iter{niter}.txt", f"logs/log_obsnoise.txt", f"logs/log_obsnoise_ps_iter{niter}.txt", f"logs/log_obsnoise_u_iter{niter}.txt", f"logs/log_obsnoise_v_iter{niter}.txt", f"logs/log_obsnoise_t_iter{niter}.txt", f"logs/log_obsnoise_q_iter{niter}.txt", f"logs/log_sysnoise_iter{niter}.txt", f"logs/trainloss_KL_iter{niter}.txt", f"logs/trainloss_integral_iter{niter}.txt", f"logs/trainloss_integral_ps_iter{niter}.txt", f"logs/trainloss_integral_u_iter{niter}.txt", f"logs/trainloss_integral_v_iter{niter}.txt", f"logs/trainloss_integral_t_iter{niter}.txt", f"logs/trainloss_integral_q_iter{niter}.txt", f"logs/testloss1_KL_iter{niter}.txt", f"logs/testloss1_integral_iter{niter}.txt", f"logs/testloss2_KL_iter{niter}.txt", f"logs/testloss2_integral_iter{niter}.txt", f"logs/gmax_iter{niter}.txt", f"logs/gmedian_iter{niter}.txt", f"logs/gmin_iter{niter}.txt"]:
        filepath = os.path.join(outdir, filename)
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"{filename} did not exist: {e}")

    # Set up asynchronous loggers for each metric.
    logger_trainloss_integral, listener_integral = setup_logger("trainloss_integral", os.path.join(outdir, f"logs/trainloss_integral_iter{niter}.txt"))
    logger_trainloss_integral_ps, listener_integral_ps = setup_logger("trainloss_integral_ps", os.path.join(outdir, f"logs/trainloss_integral_ps_iter{niter}.txt"))
    logger_trainloss_integral_u, listener_integral_u = setup_logger("trainloss_integral_u", os.path.join(outdir, f"logs/trainloss_integral_u_iter{niter}.txt"))
    logger_trainloss_integral_v, listener_integral_v = setup_logger("trainloss_integral_v", os.path.join(outdir, f"logs/trainloss_integral_v_iter{niter}.txt"))
    logger_trainloss_integral_t, listener_integral_t = setup_logger("trainloss_integral_t", os.path.join(outdir, f"logs/trainloss_integral_t_iter{niter}.txt"))
    logger_trainloss_integral_q, listener_integral_q = setup_logger("trainloss_integral_q", os.path.join(outdir, f"logs/trainloss_integral_q_iter{niter}.txt"))
    logger_trainloss_KL, listener_KL = setup_logger("trainloss_KL", os.path.join(outdir, f"logs/trainloss_KL_iter{niter}.txt"))
    logger_testloss1_integral, listener_integral_test1 = setup_logger("testloss1_integral", os.path.join(outdir, f"logs/testloss1_integral_iter{niter}.txt"))
    logger_testloss1_KL, listener_KL_test1 = setup_logger("testloss1_KL", os.path.join(outdir, f"logs/testloss1_KL_iter{niter}.txt"))
    logger_testloss2_integral, listener_integral_test2 = setup_logger("testloss2_integral", os.path.join(outdir, f"logs/testloss2_integral_iter{niter}.txt"))
    logger_testloss2_KL, listener_KL_test2 = setup_logger("testloss2_KL", os.path.join(outdir, f"logs/testloss2_KL_iter{niter}.txt"))
    logger_obsnoise, listener_obs = setup_logger("log_obsnoise", os.path.join(outdir, f"logs/log_obsnoise.txt"))
    logger_sysnoise, listener_sys = setup_logger("log_sysnoise", os.path.join(outdir, f"logs/log_sysnoise_iter{niter}.txt"))
    logger_lr, listener_lr = setup_logger("lr", os.path.join(outdir, f"logs/lr_iter{niter}.txt"))
    logger_gmax, listener_gmax = setup_logger("gmax", os.path.join(outdir, f"logs/gmax_iter{niter}.txt"))
    logger_gmedian, listener_gmedian = setup_logger("gmedian", os.path.join(outdir, f"logs/gmedian_iter{niter}.txt"))
    logger_gmin, listener_gmin = setup_logger("gmin", os.path.join(outdir, f"logs/gmin_iter{niter}.txt"))

    for i_epoch in range(num_epoch):
        print(f"{i_epoch=}")
        for i_iter, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            obs_data = batch[0].to("cuda", non_blocking=True)
            target = batch[1].to("cuda", non_blocking=True)
            obs_data = rearrange(obs_data, "bs t h latlon -> bs t (h latlon)") # (bs step 33 48*96) -> (bs step 33*48*96)
            target = rearrange(target, "bs t h latlon -> bs t (h latlon)") # (bs step 33 48*96) -> (bs step 33*48*96)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, loss_kl, loss_integral, mu_pred_list, mu_filtered_list, sigma_pred_list, sigma_filtered_list, gmax, gmedian, gmin, loss_integral_ps, loss_integral_u, loss_integral_v, loss_integral_t, loss_integral_q = model(obs_seq=obs_data, target_seq=target)

            # float16
            '''
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            '''
            # bfloat16
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            scheduler.step()
            
            logger_trainloss_integral.info(loss_integral.item())
            logger_trainloss_integral_ps.info(loss_integral_ps.item())
            logger_trainloss_integral_u.info(loss_integral_u.item())
            logger_trainloss_integral_v.info(loss_integral_v.item())
            logger_trainloss_integral_t.info(loss_integral_t.item())
            logger_trainloss_integral_q.info(loss_integral_q.item())
            logger_trainloss_KL.info(loss_kl.item())
            logger_sysnoise.info(str(model.log_Q.detach().cpu().numpy()))
            logger_lr.info(scheduler.get_last_lr()[0])
            logger_gmax.info(gmax.item())
            logger_gmedian.info(gmedian.item())
            logger_gmin.info(gmin.item())
            if i_iter % 1000 == 0:
                for var in ("ps", "u", "v", "t", "q"):
                    tensor = getattr(model, f"log_R_{var}")
                    save_path = save_obs_noise(tensor, var, i_iter, out_dir=os.path.join(outdir, "logs"))
                    logger_obsnoise.info(f"{var}: saved snapshot -> {save_path}")
                    
        # test
        print("testing...")
        torch.cuda.synchronize()
        model.eval()
        if ((i_epoch+1) % 1 == 0):
            torch.save(model.state_dict(), os.path.join(outdir, f"model_weights/model_iter{niter}_epoch{i_epoch}"))
            print("weight saved")
        testloss1_all = 0
        testloss1_kl_all = 0
        testloss1_integral_all = 0
        testloss2_all = 0
        testloss2_kl_all = 0
        testloss2_integral_all = 0
        with torch.no_grad():
            print("testing...")
            for i_iter, batch in enumerate(testloader1):
                obs_data = batch[0].to("cuda", non_blocking=True)
                target = batch[1].to("cuda", non_blocking=True)
                obs_data = rearrange(obs_data, "bs t h latlon -> bs t (h latlon)") # (bs step 33 48*96) -> (bs step 33*48*96)
                target = rearrange(target, "bs t h latlon -> bs t (h latlon)") # (bs step 33 48*96) -> (bs step 33*48*96)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    testloss1, testloss1_kl, testloss1_integral, mu_pred_list1, mu_filtered_list1, sigma_pred_list1, sigma_filtered_list1, gmax, gmedian, gmin, testloss1_integral_ps, testloss1_integral_u, testloss1_integral_v, testloss1_integral_t, testloss1_integral_q = model(obs_seq=obs_data, target_seq=target)
                    testloss1_all += testloss1.item()
                    testloss1_kl_all += testloss1_kl.item()
                    testloss1_integral_all += testloss1_integral.item()
                torch.save(mu_pred_list1.detach(), os.path.join(outdir, f"model_outputs/mu_pred_list1_iter{niter}_epoch{i_epoch}_batch{i_iter}"))
                torch.save(mu_filtered_list1.detach(), os.path.join(outdir, f"model_outputs/mu_filtered_list1_iter{niter}_epoch{i_epoch}_batch{i_iter}"))
                torch.save(sigma_pred_list1.detach(), os.path.join(outdir, f"model_outputs/sigma_pred_list1_iter{niter}_epoch{i_epoch}_batch{i_iter}"))
                torch.save(sigma_filtered_list1.detach(), os.path.join(outdir, f"model_outputs/sigma_filtered_list1_iter{niter}_epoch{i_epoch}_batch{i_iter}"))

            print("test1 ended")
            logger_testloss1_integral.info(testloss1_integral_all)
            logger_testloss1_KL.info(testloss1_kl_all)

            for i_iter, batch in enumerate(testloader2):
                obs_data = batch[0].to("cuda", non_blocking=True)
                target = batch[1].to("cuda", non_blocking=True)
                obs_data = rearrange(obs_data, "bs t h latlon -> bs t (h latlon)") # (bs step 33 48*96) -> (bs step 33*48*96)
                target = rearrange(target, "bs t h latlon -> bs t (h latlon)") # (bs step 33 48*96) -> (bs step 33*48*96)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    testloss2, testloss2_kl, testloss2_integral, mu_pred_list2, mu_filtered_list2, sigma_pred_list2, sigma_filtered_list2, gmax, gmedian, gmin, testloss2_integral_ps, testloss2_integral_u, testloss2_integral_v, testloss2_integral_t, testloss2_integral_q = model(obs_seq=obs_data, target_seq=target)
                    testloss2_all += testloss2.item()
                    testloss2_kl_all += testloss2_kl.item()
                    testloss2_integral_all += testloss2_integral.item()
                    
                torch.save(mu_pred_list2.detach(), os.path.join(outdir, f"model_outputs/mu_pred_list2_iter{niter}_epoch{i_epoch}_batch{i_iter}"))
                torch.save(mu_filtered_list2.detach(), os.path.join(outdir, f"model_outputs/mu_filtered_list2_iter{niter}_epoch{i_epoch}_batch{i_iter}"))
                torch.save(sigma_pred_list2.detach(), os.path.join(outdir, f"model_outputs/sigma_pred_list2_iter{niter}_epoch{i_epoch}_batch{i_iter}"))
                torch.save(sigma_filtered_list2.detach(), os.path.join(outdir, f"model_outputs/sigma_filtered_list2_iter{niter}_epoch{i_epoch}_batch{i_iter}"))

            logger_testloss2_integral.info(testloss2_integral_all)
            logger_testloss2_KL.info(testloss2_kl_all)
        
        #send_epoch_email(loss_integral.item(), loss_kl.item(), testloss1_integral_all, testloss1_kl_all, testloss2_integral_all, testloss2_kl_all, model.log_R.detach().cpu().numpy(), model.log_Q.detach().cpu().numpy(), i_epoch, from_email, to_email, smtp_user, smtp_password)
        print(f"Epoch {i_epoch} complete")
        
        model.train()

    # Stop all the listeners to flush and close the logging threads.
    listener_integral.stop()
    listener_KL.stop()
    listener_integral_test1.stop()
    listener_integral_test2.stop()
    listener_KL_test1.stop()
    listener_KL_test2.stop()
    listener_obs.stop()
    listener_sys.stop()
    listener_lr.stop()
    listener_gmax.stop()
    listener_gmedian.stop()
    listener_gmin.stop()

if __name__ == "__main__":
    main()
