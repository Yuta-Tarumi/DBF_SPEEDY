import torch
import torch.nn as nn
from einops import rearrange
#from torch.distributions import Normal, MultivariateNormal, kl_divergence
from DBF.model.speedy import TransformerToLatent_sparse2_fG, TransformerToLatent_sparser_fG, TransformerToLatent_sparsest_fG, TransformerToLatent_sparse2_fG_MLPMixerLike, Pixelshuffle, TransformerDecoder, TransformerDecoderNoMixing, TransformerDecoder_MLPMixerLike
                        
def inv_2x2(X):
    a = X[..., 0, 0]  # shape (B, C)
    b = X[..., 0, 1]
    c = X[..., 1, 0]
    d = X[..., 1, 1]
    #print(f"{a.device=}")
    det = a * d - b * c
    inv_det = 1.0 / det  # make sure det is not zero (or add a small epsilon)

    # Build the inverse for each 2x2 block:
    # The new tensor will have the same batch shape (B, C, 2, 2)
    X_inv = torch.stack([
        torch.stack([ d, -b ], dim=-1),
        torch.stack([-c,  a ], dim=-1)
    ], dim=-2) * inv_det.unsqueeze(-1).unsqueeze(-1)
    return X_inv

def prod2x2(A, B):
    a00 = A[..., 0, 0]
    a01 = A[..., 0, 1]
    a10 = A[..., 1, 0]
    a11 = A[..., 1, 1]

    b00 = B[..., 0, 0]
    b01 = B[..., 0, 1]
    b10 = B[..., 1, 0]
    b11 = B[..., 1, 1]

    # Compute the elements of the product C = A * B
    c00 = a00 * b00 + a01 * b10
    c01 = a00 * b01 + a01 * b11
    c10 = a10 * b00 + a11 * b10
    c11 = a10 * b01 + a11 * b11

    # Stack to recover the (batch, 2, 2) shape
    C = torch.stack([
        torch.stack([c00, c01], dim=-1),
        torch.stack([c10, c11], dim=-1)
    ], dim=-2)
    return C

def prod2x1(M, V):
    # Assume T is of shape (..., 2, 2) and V is of shape (..., 2)
    
    # Extract the elements of each 2×2 block
    a = M[..., 0, 0]
    b = M[..., 0, 1]
    c = M[..., 1, 0]
    d = M[..., 1, 1]
    
    # Extract the vector components
    x = V[..., 0]
    y = V[..., 1]

    # Compute the resulting components using the 2×2 matrix-vector formula:
    # [a b; c d] * [x; y] = [a*x + b*y; c*x + d*y]
    r0 = a * x + b * y
    r1 = c * x + d * y

    # Stack the results to form a (..., 2) tensor
    result = torch.stack([r0, r1], dim=-1)
    return result
    
def compute_K_sigma_block_diag(lambdas: torch.Tensor, num_real=0, num_complex=1):
    z_dim = num_real + num_complex * 2
    mus = 0.01 * lambdas[0::2]
    omegas = lambdas[1::2]
    cosine = torch.cos(omegas).view(num_complex, 1)
    sine = torch.sin(omegas).view(num_complex, 1)
    Jordan_block = torch.exp(mus).view(num_complex, 1, 1) * torch.cat((cosine, -sine, sine, cosine), dim=1).view(num_complex, 2, 2)
    return Jordan_block

def kl_divergence_gaussians(mu_p, cov_p, mu_q, cov_q):
    inv_cov_q, _ = torch.linalg.inv_ex(cov_q)  # (batch, dim, dim)
    diff = (mu_q - mu_p)  # (batch, dim, 1)
    # Compute trace of (inv_cov_q @ cov_p) for each batch element
    trace_term = torch.einsum("btzij,btzij->btz", inv_cov_q, cov_p)
    # Quadratic term: (mu_q - mu_p)^T inv_cov_q (mu_q - mu_p)
    quad_term = torch.einsum("btzi,btzij,btzj->btz", diff, inv_cov_q, diff)
    log_det_cov_p = torch.logdet(cov_p)
    log_det_cov_q = torch.logdet(cov_q)
    d = mu_p.shape[-1]
    kl = 0.5 * (trace_term + quad_term - d + (log_det_cov_q - log_det_cov_p))
    return kl  # shape (batch,)

def sample_gaussian(mu, cov, num_samples):
    # mu: shape (batch, dim)
    # cov: shape (batch, dim, dim)
    batch, T, z1, dim = mu.shape
    # Compute lower-triangular Cholesky factor L (assumes cov is PD)
    L, _ = torch.linalg.cholesky_ex(cov)  # shape (batch, dim, dim)
    # Sample epsilon ~ N(0, I) with shape (num_samples, batch, dim, 1)
    eps = torch.randn(num_samples, batch, T, z1, dim, 1, device=mu.device)
    # Reshape mu to (1, batch, dim, 1) to broadcast properly
    mu_ = mu.unsqueeze(0).unsqueeze(-1)
    sample = mu_ + torch.matmul(L.unsqueeze(0), eps)  # shape (num_samples, batch, dim, 1)
    sample = sample.squeeze(-1)  # now shape (num_samples, batch, dim)
    return sample  # for num_samples=1, sample[0] gives (batch, dim)

def normal_log_prob(x, mean, log_std):
    # x, mean, log_std: same shape, for independent normals per element
    # Compute element-wise log probability
    # Note: log_std is log(s) so variance = exp(2*log_std)
    var = torch.exp(2 * log_std)
    log_prob = -0.5 * (torch.log(torch.tensor(2 * torch.pi)) + 2 * log_std + ((x - mean) ** 2) / var)
    return log_prob  # then you can sum over dimensions if needed

def gamma_log_prob(x, k, theta, eps=1e-7):
    """
        Element-wise log p_Gamma(x | k, θ) with safe clamps.
        x, k, θ broadcast to the same shape.
    """
    x = x.clamp_min(eps)                        # avoid log(0) / div-0
    k = k.clamp_min(eps)
    theta = theta.clamp_min(eps)
    return (k - 1) * torch.log(x) - x / theta - k * torch.log(theta) - torch.lgamma(k)

class DeepBayesianFilterBlockDiag(nn.Module):
    def __init__(self, latent_dim, simulation_dimensions, model_seed, arch="baseline", dropout=0.0, q_distribution="Gaussian"):
        torch.manual_seed(model_seed)
        super().__init__()
        assert q_distribution in ["Gaussian", "Gamma"], "q_distribution should be either Gamma or Gaussian."
        self.q_distribution = q_distribution
        print(f"{self.q_distribution=}")
        assert latent_dim % 2 == 0, "latent_dim must be even for 2x2 block structure."
        self.latent_dim = latent_dim
        self.D_norm = (1+8*3)*48*96 # ps(1), u(8), v(8), t(8) for all 48*96 lat&lons
        self.num_blocks = latent_dim // 2
        self.lambdas = nn.Parameter(0.01*torch.randn(latent_dim))
        self.A_block = compute_K_sigma_block_diag(
            self.lambdas, num_real=0, num_complex=self.num_blocks
        )
        self.log_Q = nn.Parameter(torch.tensor(0.0))
        init_val = -1.0
        L, H, W = simulation_dimensions
        #self.log_R = nn.Parameter(torch.tensor(-1.0))
        self.log_R_ps = nn.Parameter(torch.full((H * W, ), init_val))  # surface
        self.log_R_u = nn.Parameter(torch.full((L * H * W, ), init_val))  # volumetric
        self.log_R_v = nn.Parameter(torch.full((L * H * W, ), init_val))
        self.log_R_t = nn.Parameter(torch.full((L * H * W, ), init_val))
        self.log_R_q = nn.Parameter(torch.full((L * H * W, ), init_val))
        
        self.init_mu = torch.zeros(self.num_blocks, 2, device="cuda")
        self.init_sigma = 100*torch.eye(2, device="cuda").unsqueeze(0).repeat(self.num_blocks, 1, 1)

        if arch == "baseline":
            self.encoder = TransformerToLatent_sparse2_fG(
                latent_dim=self.latent_dim,
            )
            self.decoder = Pixelshuffle(
                latent_dim=self.latent_dim,
                out_channels=1,
                height=8,
                latitude=48,
                longitude=96,
            )
        elif arch == "decoder_transformer_sparse":
            print("patch_size=3")
            self.encoder = TransformerToLatent_sparse2_fG(
                latent_dim=self.latent_dim,
                dim_intermediate=32,
                embed_dim=1024,
                num_heads=8,
                num_layers=8,
                dropout=dropout
            )
            self.decoder = TransformerDecoder(
                latent_dim=self.latent_dim,
                fc_middle_dim=9216,
                out_channels=33,
                token_dim=1024,
                height=33,
                latitude=48,
                longitude=96,
                num_layers=8,
                ffn_hidden_factor=2,
                num_attention_heads=8,
                patch_size=3,
                dropout=dropout
            )
        elif arch == "decoder_transformer2":
            print("patch_size=4")
            self.encoder = TransformerToLatent_sparse2_fG_MLPMixerLike(
                latent_dim=self.latent_dim,
                dim_intermediate=64,
                embed_dim=1024,
                num_heads=8,
                num_layers=8,
                num_MLPMixer_layers=8,
                dropout=dropout
            )
            self.decoder = TransformerDecoder_MLPMixerLike(
                latent_dim=self.latent_dim,
                fc_middle_dim=18432,
                out_channels=33,
                token_dim=1024,
                height=33,
                latitude=48,
                longitude=96,
                num_layers=8,
                num_MLPMixer_layers=8,
                ffn_hidden_factor=2,
                num_attention_heads=8,
                patch_size=4,
                dropout=dropout
            )
        elif arch == "decoder_largetransformer":
            self.encoder = TransformerToLatent_sparse2_fG(
                latent_dim=self.latent_dim,
                dim_intermediate=32,
                embed_dim=2048,
                num_heads=8,
                num_layers=8,
                dropout=dropout
            )
            self.decoder = TransformerDecoder(
                latent_dim=self.latent_dim,
                fc_middle_dim=9216,
                out_channels=33,
                token_dim=2048,
                height=33,
                latitude=48,
                longitude=96,
                num_layers=8,
                ffn_hidden_factor=2,
                num_attention_heads=8,
                patch_size=4,
                dropout=dropout
            )
        elif arch == "decoder_transformer_sparser":
            self.encoder = TransformerToLatent_sparser_fG(
                latent_dim=self.latent_dim,
                dim_intermediate=64,
                embed_dim=1024,
                num_heads=8,
                num_layers=12,
                dropout=dropout
            )
            self.decoder = TransformerDecoder(
                latent_dim=self.latent_dim,
                fc_middle_dim=9216,
                out_channels=33,
                token_dim=1024,
                height=33,
                latitude=48,
                longitude=96,
                num_layers=12,
                ffn_hidden_factor=2,
                num_attention_heads=8,
                patch_size=4,
                dropout=dropout
            )
        elif arch == "decoder_transformer_sparsest":
            print("patch_size=3")
            self.encoder = TransformerToLatent_sparsest_fG(
                latent_dim=self.latent_dim,
                dim_intermediate=64,
                embed_dim=1024,
                num_heads=8,
                num_layers=12,
                dropout=dropout
            )
            self.decoder = TransformerDecoder(
                latent_dim=self.latent_dim,
                fc_middle_dim=9216,
                out_channels=33,
                token_dim=1024,
                height=33,
                latitude=48,
                longitude=96,
                num_layers=12,
                ffn_hidden_factor=2,
                num_attention_heads=8,
                patch_size=3,
                dropout=dropout
            )
        else:
            print(f"{arch=} undefined")
            return 0

        print(f"{self.encoder=}")
        print(f"{self.decoder=}")
        if arch in ["decoder_transformer", "decoder_largetransformer"]:
            print(f"{sum(p.numel() for p in self.encoder.fc_mix.parameters() if p.requires_grad)=}")
            print(f"{sum(p.numel() for p in self.decoder.fc_mix.parameters() if p.requires_grad)=}")
        print(f"{sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)=}")
        print(f"{sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)=}")

    def _concat_log_R(self):
        return torch.cat([
            self.log_R_ps,
            self.log_R_u,
            self.log_R_v,
            self.log_R_t,
            self.log_R_q,
        ], dim=0)
        
    def compute_loss(
        self,
        mu_filtered_list,
        sigma_filtered_list,
        mu_pred_list,
        sigma_pred_list,
        batch_size,
        target,
        T
    ):
        loss_kl = torch.sum(kl_divergence_gaussians(mu_filtered_list, sigma_filtered_list,
                                              mu_pred_list, sigma_pred_list)) / batch_size
        num_samples = 1
        X_samples = sample_gaussian(mu_filtered_list, sigma_filtered_list, num_samples=num_samples)
        X_samples_flat = rearrange(X_samples, "n b t z i -> (n b t) (z i)")
        reparametrized_flat = self.decoder(X_samples_flat)
        reparametrized = rearrange(reparametrized_flat, "(n b t) x -> n b t x", n=num_samples, b=batch_size, t=T)

        # Compute log-likelihood manually
        log_R_full = self._concat_log_R()[None,None,None,:]
        p_Y_norm_log = normal_log_prob(target[..., :self.D_norm], reparametrized[..., :self.D_norm], log_R_full[..., :self.D_norm])
        ### q distribution gamma or normal? ###
        if self.q_distribution == "Gaussian":
            p_Y_q_log = normal_log_prob(target[..., self.D_norm:], reparametrized[..., self.D_norm:], log_R_full[..., self.D_norm:])
        elif self.q_distribution == "Gamma":
            q_k = reparametrized[..., self.D_norm:].exp() # to ensure positivity
            q_theta = torch.exp(log_R_full[..., self.D_norm:])
            p_Y_q_log = gamma_log_prob(target[..., self.D_norm:], q_k, q_theta)
        else:
            print("not supported")
            return 0
        ### q distribution gamma or normal? ###
        p_Y_given_X_log = torch.cat([p_Y_norm_log, p_Y_q_log], dim=-1)

        # loss for each term
        loss_integral_ps = -torch.sum(p_Y_given_X_log[..., :48*96]) / (batch_size * num_samples)
        loss_integral_u = -torch.sum(p_Y_given_X_log[..., 48*96:9*48*96]) / (batch_size * num_samples)
        loss_integral_v = -torch.sum(p_Y_given_X_log[..., 9*48*96:17*48*96]) / (batch_size * num_samples)
        loss_integral_t = -torch.sum(p_Y_given_X_log[..., 17*48*96:25*48*96]) / (batch_size * num_samples)
        #loss_integral_phi = -torch.sum(p_Y_given_X_log[..., 25*48*96:33*48*96]) / (batch_size * num_samples)
        loss_integral_q = -torch.sum(p_Y_given_X_log[..., 25*48*96:33*48*96]) / (batch_size * num_samples)
        
        loss_integral = -torch.sum(p_Y_given_X_log) / (batch_size * num_samples)
        total_loss = loss_integral + loss_kl
        # Legacy code
        '''
        p = MultivariateNormal(loc=mu_filtered_list, covariance_matrix=sigma_filtered_list)
        q = MultivariateNormal(loc=mu_pred_list, covariance_matrix=sigma_pred_list)
        # Compute the KL divergence loss
        loss_kl2 = torch.sum(kl_divergence(p, q)) / batch_size
        print(f"{loss_kl2=}")
        # Create the two distributions
        p = MultivariateNormal(loc=mu_filtered_list, covariance_matrix=sigma_filtered_list)
        q = MultivariateNormal(loc=mu_pred_list, covariance_matrix=sigma_pred_list)
        # Compute the KL divergence loss
        loss_kl = torch.sum(kl_divergence(p, q)) / batch_size
        # Integral loss computation
        num_samples = 1
        X_samples = p.rsample((num_samples,)).cuda()
        X_samples_flat = rearrange(X_samples, "n b t z i -> (n b t) (z i)")
        reparametrized_flat = self.decoder(X_samples_flat)
        reparametrized = rearrange(reparametrized_flat, "(n b t) x -> n b t x", n=num_samples, b=batch_size, t=T)
        
        # Create the likelihood distribution
        p_Y_given_X = Normal(reparametrized, torch.exp(self.log_R))
        target_batched = target.unsqueeze(0)
        
        loss_integral = -torch.sum(p_Y_given_X.log_prob(target_batched)) / (batch_size * num_samples)
        
        # Total loss is the sum of the two components
        total_loss = loss_integral + loss_kl
        '''
        return total_loss, loss_kl, loss_integral, loss_integral_ps.detach(), loss_integral_u.detach(), loss_integral_v.detach(), loss_integral_t.detach(), loss_integral_q.detach()
        
    def forward(self, obs_seq, target_seq):
        """
        obs_seq: tensor of shape (batch, T, obs_dim)
        Returns: total loss and the sequence of filtering posterior means (shape: batch x T x latent_dim)
        """
        batch_size, T, _ = obs_seq.shape
        device = obs_seq.device

        # initialize
        init_mu = self.init_mu.unsqueeze(0).expand(batch_size, self.num_blocks, 2)
        init_sigma = self.init_sigma.unsqueeze(0).expand(batch_size, self.num_blocks, 2, 2)

        mu_pred_list = torch.empty(batch_size, T, self.num_blocks, 2, device="cuda")
        mu_filtered_list = torch.empty(batch_size, T, self.num_blocks, 2, device="cuda")
        sigma_pred_list = torch.empty(batch_size, T, self.num_blocks, 2, 2, device="cuda")
        sigma_filtered_list = torch.empty(batch_size, T, self.num_blocks, 2, 2, device="cuda")
        mu_pred = init_mu 
        sigma_pred = init_sigma
        mu_pred_list[:, 0] = mu_pred
        sigma_pred_list[:, 0] = sigma_pred
        
        # compute A_block here
        self.Q_tensor = torch.exp(self.log_Q) * torch.eye(2, device="cuda")
        self.A_block = compute_K_sigma_block_diag(
            self.lambdas, num_real=0, num_complex=self.num_blocks
        )
        # observation processing
        maximum_g = 100
        obs_batched = rearrange(obs_seq, "b t x -> (b t) x")
        enc_out_batched = self.encoder(obs_batched)
        enc_out = rearrange(enc_out_batched, "(b t) x -> b t x", t=T)
        f_enc_all = enc_out[:, :, :self.latent_dim].reshape(batch_size, T, self.num_blocks, 2)
        g_enc_all = maximum_g*torch.tanh(enc_out[:, :, self.latent_dim:]**2/maximum_g).reshape(batch_size, T, self.num_blocks, 2) # should contain g^-
        gf_all = g_enc_all*f_enc_all
        
        for t in range(T):
            # update step
            f_enc = f_enc_all[:, t]
            g_enc = g_enc_all[:, t]
            
            sigma_pred_inv, _ = torch.linalg.inv_ex(sigma_pred)
            sigma_filtered, _ = torch.linalg.inv_ex(sigma_pred_inv + torch.diag_embed(g_enc))
            sigma_filtered = 0.5*(sigma_filtered + sigma_filtered.transpose(-2, -1)) # ensure symmetricity
            mu_filtered = torch.einsum("bzij,bzjk,bzk->bzi", sigma_filtered, sigma_pred_inv, mu_pred) + gf_all[:, t]
            mu_pred = torch.einsum("zxy,bzy->bzx", self.A_block, mu_filtered)
            sigma_pred = torch.einsum("zij,bzjk,zlk->bzil", self.A_block, sigma_filtered, self.A_block)
            sigma_pred = 0.5*(sigma_pred + sigma_pred.transpose(-2, -1)) # ensure symmetricity
            sigma_pred = sigma_pred + self.Q_tensor
            # save tensors for loss computation
            mu_filtered_list[:, t] = mu_filtered
            sigma_filtered_list[:, t] = sigma_filtered
            if t+1 < T:
                mu_pred_list[:, t+1] = mu_pred
                sigma_pred_list[:, t+1] = sigma_pred
        
        # loss computation
        loss, loss_kl, loss_integral, loss_integral_ps, loss_integral_u, loss_integral_v, loss_integral_t, loss_integral_q = self.compute_loss(
            mu_filtered_list=mu_filtered_list,
            sigma_filtered_list=sigma_filtered_list,
            mu_pred_list=mu_pred_list,
            sigma_pred_list=sigma_pred_list,
            batch_size=batch_size,
            target=target_seq,
            T=T
        )
        return loss, loss_kl, loss_integral, mu_pred_list, mu_filtered_list, sigma_pred_list, sigma_filtered_list, torch.max(g_enc_all), torch.median(g_enc_all), torch.min(g_enc_all), loss_integral_ps, loss_integral_u, loss_integral_v, loss_integral_t, loss_integral_q
