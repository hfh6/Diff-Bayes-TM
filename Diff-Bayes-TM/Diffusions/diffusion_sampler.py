import torch

from tqdm.auto import tqdm
from Diffusions.diffusion_model import GaussianDiffusion
from Diffusions.useful_functions import mat_by_vec, extract


class DiffusionSampler(object):

    def __init__(self, unconditional_diffusion_model: GaussianDiffusion):
        super().__init__()
        self.diff = unconditional_diffusion_model
        self.num_timesteps = self.diff.num_timesteps
        self.self_condition = self.diff.self_condition
        self.is_ddim_sampling = self.diff.is_ddim_sampling
        self.ddim_sampling_eta = self.diff.ddim_sampling_eta
        self.sampling_timesteps = self.diff.sampling_timesteps
        self.seq_length, self.feature_size = self.diff.seq_length, self.diff.feature_size

    @torch.no_grad()
    def uncond_sample(self, batch_size: int = 16):
        seq_length, feature_size = self.seq_length, self.feature_size
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        tm = sample_fn((batch_size, seq_length, feature_size))
        return tm

    def ddim_sample(self, shape: tuple, clip_denoised: bool = True):
        sampling_timesteps, eta = self.sampling_timesteps, self.ddim_sampling_eta
        batch, device, total_timesteps = shape[0], self.diff.betas.device, self.num_timesteps

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        tm = torch.randn(shape, device=device)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.diff.model_predictions(tm, time_cond, self_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                tm = x_start
                continue

            alpha = self.diff.alphas_cumprod[time]
            alpha_next = self.diff.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(tm)

            tm = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return tm

    def p_sample_loop(self, shape: tuple):
        x_start = None
        device = self.diff.betas.device
        tm = torch.randn(shape, device=device)

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                      total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            tm, x_start = self.p_sample(tm, t, self_cond)

        return tm

    def p_sample(
            self,
            x: torch.Tensor,
            t: int,
            x_self_cond: torch.Tensor = None,
            clip_denoised: bool = True
    ):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = \
            self.diff.p_mean_variance(x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_tm = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_tm, x_start

    def cond_sample(
            self,
            observation: torch.Tensor,
            problem: str,
            sample_kwargs: dict = None
    ):
        batch_size = observation.shape[0]
        feature_size, channels = self.feature_size, self.seq_length
        shape = (batch_size, channels, feature_size)

        style = sample_kwargs.get('style', 'monte_carlo')

        if problem in ["TMC", "tmc"]:
            if style == 'monte_carlo':
                raise NotImplementedError("Monte Carlo sampling is not implemented yet.")
            else:
                tmc_fn = self.TMC if not self.is_ddim_sampling else self.fast_TMC
                tm = tmc_fn(shape, y=observation, sample_kwargs=sample_kwargs, clip_denoised=True)
        elif problem in ["NT", "nt"]:
            if style == 'monte_carlo':
                raise NotImplementedError("Monte Carlo sampling is not implemented yet.")
            else:
                nt_fn = self.NT if not self.is_ddim_sampling else self.fast_NT
                tm = nt_fn(shape, y=observation, **sample_kwargs, clip_denoised=True)
        else:
            raise ValueError("problem must be either TMC or NT")

        return tm

    @torch.no_grad()
    def sample_w_monte(
            self,
            shape: tuple,
            y: torch.Tensor,
            problem: str,
            partial_mask: torch.Tensor = None,
            initial_num_particles: int = 64,
            log_scale: float = 10 ** 7,
            A: torch.Tensor = None,
            clip_denoised: bool = True
    ):
        batch_size = shape[0]
        device = self.diff.betas.device

        batch_initial_particles = torch.randn(size=(initial_num_particles, batch_size, shape[1], shape[2]),
                                              device=device)

        if problem in ["TMC", "tmc"]:
            tms = self.tmc_w_monte(y, batch_initial_particles, partial_mask, clip_denoised, log_scale) if \
                not self.is_ddim_sampling else \
                self.fast_tmc_w_monte(y, batch_initial_particles, partial_mask, clip_denoised, log_scale)
        elif problem in ["NT", "nt"]:
            tms = self.nt_w_monte(y, batch_initial_particles, A, clip_denoised, log_scale) if \
                not self.is_ddim_sampling else \
                self.fast_nt_w_monte(y, batch_initial_particles, A, clip_denoised, log_scale)
        else:
            raise ValueError("problem must be either TMC or NT")

        return tms

    def tmc_w_monte(
            self,
            y: torch.Tensor,
            initial_particles: torch.Tensor,
            partial_mask: torch.Tensor,
            clip_denoised: bool = True,
            log_scale: float = 10 ** 7
    ):
        device = initial_particles.device
        num_particles = initial_particles.shape[0]
        particles = initial_particles
        B, T, D = particles.shape[1], particles.shape[2], particles.shape[3]

        active_mask = (partial_mask > 0).reshape(partial_mask.shape).to(device)
        inactive_mask = (partial_mask <= 0).reshape(partial_mask.shape).to(device)

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='Monte Carlo guided sample time step',
                      total=self.num_timesteps):

            y_times = torch.full((y.shape[0],), t, device=device, dtype=torch.long)
            mean_y_t = extract(self.diff.sqrt_alphas_cumprod, y_times, y.shape) * y

            x = particles.reshape(-1, particles.shape[-2], particles.shape[-1])
            batched_times = torch.full((num_particles * particles.shape[1],), t, device=device, dtype=torch.long)
            preds = self.diff.model_predictions(x, batched_times, clip_x_start=clip_denoised)
            x_start = preds.pred_x_start.reshape(particles.shape)

            log_likelihood = self.tmc_likelihood(x=x_start, mean=y, mask=active_mask,
                                                 scale=log_scale)

            if t > self.num_timesteps * 0.9:
                num_particles = 8
            elif t > self.num_timesteps * 0.8:
                num_particles = 4
            else:
                num_particles = 2

            j_idx = torch.arange(B).repeat(num_particles)

            dist = torch.distributions.Categorical(logits=-log_likelihood)
            sampled_index = dist.sample((num_particles,))

            new_particles = particles[sampled_index.reshape(-1), j_idx]
            x_start = x_start[sampled_index.reshape(-1), j_idx]

            new_batched_times = torch.full((num_particles * particles.shape[1],), t, device=device, dtype=torch.long)

            model_mean, _, model_log_variance = self.diff.q_posterior(x_start=x_start, x_t=new_particles,
                                                                      t=new_batched_times)
            noise = torch.randn_like(model_mean) if t > 0 else 0.
            sigma = (0.5 * model_log_variance).exp()
            pred_x = model_mean + sigma * noise
            pred_x = pred_x.reshape(num_particles, B, T, D)

            kappa = (sigma ** 2 / (sigma ** 2 + 1 - self.diff.alphas[t])).reshape(num_particles, B, 1, 1)
            kappa = kappa.repeat(1, 1, T, D)[:, active_mask]
            # kappa = 0.5

            particles = new_particles.reshape(pred_x.shape)

            particles[:, active_mask] = kappa * pred_x[:, active_mask] + (1 - kappa) * mean_y_t[active_mask]
            particles[:, inactive_mask] = pred_x[:, inactive_mask]

        log_likelihood = self.tmc_likelihood(x=pred_x, mean=y, mask=active_mask,
                                             scale=log_scale)

        dist = torch.distributions.Categorical(logits=-log_likelihood)
        sampled_index = dist.sample((1,))

        sampled_index = sampled_index.repeat(1, num_particles)
        tms = particles[sampled_index.reshape(-1), j_idx].reshape(particles.shape)[0, :, :, :]
        tms[active_mask] = y[active_mask]
        return tms

    def fast_tmc_w_monte(
            self,
            y: torch.Tensor,
            initial_particles: torch.Tensor,
            partial_mask: torch.Tensor,
            clip_denoised: bool = True,
            log_scale: float = 10 ** 7
    ):
        device = initial_particles.device
        num_particles = initial_particles.shape[0]
        particles = initial_particles
        B, T, D = particles.shape[1], particles.shape[2], particles.shape[3]

        total_timesteps, sampling_timesteps = self.num_timesteps, self.sampling_timesteps
        eta = self.ddim_sampling_eta if self.ddim_sampling_eta != 0 else 0.5
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        active_mask = (partial_mask > 0).reshape(partial_mask.shape).to(device)

        for time, time_next in tqdm(time_pairs, desc='Monte Carlo guided sample time step'):

            x = particles.reshape(-1, particles.shape[-2], particles.shape[-1])
            batched_times = torch.full((num_particles * particles.shape[1],), time, device=device, dtype=torch.long)

            preds = self.diff.model_predictions(x, batched_times, clip_x_start=clip_denoised)
            x_start = preds.pred_x_start.reshape(particles.shape)
            pred_noise = preds.pred_noise.reshape(particles.shape)

            log_likelihood = self.tmc_likelihood(x=x_start, mean=y, mask=active_mask,
                                                 scale=log_scale)

            if time > self.num_timesteps * 0.9:
                num_particles = 8
            elif time > self.num_timesteps * 0.8:
                num_particles = 4
            else:
                num_particles = 2

            j_idx = torch.arange(B).repeat(num_particles)

            dist = torch.distributions.Categorical(logits=-log_likelihood)
            sampled_index = dist.sample((num_particles,))

            x_start = x_start[sampled_index.reshape(-1), j_idx]
            pred_noise = pred_noise[sampled_index.reshape(-1), j_idx]

            if time_next < 0:
                pred_x = x_start.reshape(num_particles, B, T, D)
                particles = x_start.reshape(num_particles, B, T, D)
                continue

            alpha = self.diff.alphas_cumprod[time]
            alpha_next = self.diff.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(pred_mean)
            pred_x = pred_mean + sigma * noise
            pred_x = pred_x.reshape(num_particles, B, T, D)

            y_times = torch.full((y.shape[0],), time_next, device=device, dtype=torch.long)
            mean_y_t = extract(self.diff.sqrt_alphas_cumprod, y_times, y.shape) * y

            kappa = (sigma ** 2 / (sigma ** 2 + 1 - self.diff.alphas[time_next]))

            particles = pred_x
            particles[:, active_mask] = kappa * pred_x[:, active_mask] + (1 - kappa) * mean_y_t[active_mask]

        log_likelihood = self.tmc_likelihood(x=pred_x, mean=y, mask=active_mask,
                                             scale=log_scale)

        dist = torch.distributions.Categorical(logits=-log_likelihood)
        sampled_index = dist.sample((1,))

        sampled_index = sampled_index.repeat(1, num_particles)
        tms = particles[sampled_index.reshape(-1), j_idx].reshape(particles.shape)[0, :, :, :]
        tms[active_mask] = y[active_mask]
        return tms

    def nt_w_monte(
            self,
            y: torch.Tensor,
            initial_particles: torch.Tensor,
            A: torch.Tensor,
            clip_denoised: bool = True,
            log_scale: float = 10 ** 7
    ):
        device = initial_particles.device
        num_particles = initial_particles.shape[0]
        particles = initial_particles
        B, T, D = particles.shape[1], particles.shape[2], particles.shape[3]

        A_pinv = torch.linalg.pinv(A).to(device)

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='Monte Carlo guided sample time step',
                      total=self.num_timesteps):

            x = particles.reshape(-1, particles.shape[-2], particles.shape[-1])
            batched_times = torch.full((num_particles * particles.shape[1],), t, device=device, dtype=torch.long)
            preds = self.diff.model_predictions(x, batched_times, clip_x_start=clip_denoised)
            x_start = preds.pred_x_start.reshape(particles.shape)

            log_likelihood = self.nt_likelihood(x_start, y, A, scale=log_scale)

            if t > self.num_timesteps * 0.9:
                num_particles = 8
            elif t > self.num_timesteps * 0.8:
                num_particles = 4
            else:
                num_particles = 2

            j_idx = torch.arange(B).repeat(num_particles)

            dist = torch.distributions.Categorical(logits=-log_likelihood)
            sampled_index = dist.sample((num_particles,))

            new_particles = particles[sampled_index.reshape(-1), j_idx]
            x_start = x_start[sampled_index.reshape(-1), j_idx]

            new_batched_times = torch.full((num_particles * particles.shape[1],), t, device=device, dtype=torch.long)

            model_mean, _, model_log_variance = self.diff.q_posterior(x_start=x_start, x_t=new_particles,
                                                                      t=new_batched_times)
            noise = torch.randn_like(model_mean) if t > 0 else 0.
            sigma = (0.5 * model_log_variance).exp()
            pred_x = model_mean + sigma * noise
            pred_x = pred_x.reshape(num_particles, B, T, D)

            if t < self.num_timesteps * 0.05:
                particles = pred_x

            else:
                kappa = (sigma ** 2 / (sigma ** 2 + 1 - self.diff.alphas[t])).reshape(num_particles, B, 1, 1)
                kappa = kappa.repeat(1, 1, T, D)

                error = y - mat_by_vec(A, x_start.reshape(pred_x.shape))
                particles = pred_x + (1 - kappa) * self.diff.sqrt_alphas_cumprod[t] * mat_by_vec(A_pinv, error)

        log_likelihood = self.nt_likelihood(pred_x, y, A, scale=log_scale)

        dist = torch.distributions.Categorical(logits=-log_likelihood)
        sampled_index = dist.sample((1,))

        sampled_index = sampled_index.repeat(1, num_particles)
        tms = particles[sampled_index.reshape(-1), j_idx].reshape(particles.shape)[0, :, :, :]
        return tms

    def fast_nt_w_monte(
            self,
            y: torch.Tensor,
            initial_particles: torch.Tensor,
            A: torch.Tensor,
            clip_denoised: bool = True,
            log_scale: float = 10 ** 7
    ):
        device = initial_particles.device
        num_particles = initial_particles.shape[0]
        particles = initial_particles
        B, T, D = particles.shape[1], particles.shape[2], particles.shape[3]

        A_pinv = torch.linalg.pinv(A).to(device)

        total_timesteps, sampling_timesteps = self.num_timesteps, self.sampling_timesteps
        eta = self.ddim_sampling_eta if self.ddim_sampling_eta != 0 else 0.5
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        for time, time_next in tqdm(time_pairs, desc='Monte Carlo guided sample time step'):

            x = particles.reshape(-1, particles.shape[-2], particles.shape[-1])
            batched_times = torch.full((num_particles * particles.shape[1],), time, device=device, dtype=torch.long)

            preds = self.diff.model_predictions(x, batched_times, clip_x_start=clip_denoised)
            x_start = preds.pred_x_start.reshape(particles.shape)
            pred_noise = preds.pred_noise.reshape(particles.shape)

            log_likelihood = self.nt_likelihood(x_start, y, A, scale=log_scale)

            if time > self.num_timesteps * 0.9:
                num_particles = 8
            elif time > self.num_timesteps * 0.8:
                num_particles = 4
            else:
                num_particles = 2

            j_idx = torch.arange(B).repeat(num_particles)

            dist = torch.distributions.Categorical(logits=-log_likelihood)
            sampled_index = dist.sample((num_particles,))

            x_start = x_start[sampled_index.reshape(-1), j_idx]
            pred_noise = pred_noise[sampled_index.reshape(-1), j_idx]

            if time_next < 0:
                pred_x = x_start.reshape(num_particles, B, T, D)
                particles = x_start.reshape(num_particles, B, T, D)
                continue

            alpha = self.diff.alphas_cumprod[time]
            alpha_next = self.diff.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(pred_mean)
            pred_x = pred_mean + sigma * noise
            pred_x = pred_x.reshape(num_particles, B, T, D)

            if time < self.num_timesteps * 0.05:
                particles = pred_x

            else:
                kappa = (sigma ** 2 / (sigma ** 2 + 1 - self.diff.alphas[time_next]))

                error = y - mat_by_vec(A, x_start.reshape(pred_x.shape))
                particles = pred_x + (1 - kappa) * self.diff.sqrt_alphas_cumprod[time_next] * mat_by_vec(A_pinv, error)

        log_likelihood = self.nt_likelihood(pred_x, y, A, scale=log_scale)

        dist = torch.distributions.Categorical(logits=-log_likelihood)
        sampled_index = dist.sample((1,))

        sampled_index = sampled_index.repeat(1, num_particles)
        tms = particles[sampled_index.reshape(-1), j_idx].reshape(particles.shape)[0, :, :, :]
        return tms

    @staticmethod
    def tmc_likelihood(x, mean, mask, scale=1e7):

        mean = mean.unsqueeze(0)

        mask = mask.unsqueeze(0)

        masked_diff = scale * torch.abs(x - mean)
        masked_diff = masked_diff * mask

        valid_counts = mask.sum(dim=(-2, -1))

        likelihood = masked_diff.sum(dim=(-2, -1)) / valid_counts.clamp(min=1e-5)

        return likelihood.transpose(0, 1)

    @staticmethod
    def nt_likelihood(x, mean, A, scale=1e7):

        mean = mean.unsqueeze(0)

        aggregated_diff = scale * torch.abs(mat_by_vec(A, x) - mean)

        likelihood = aggregated_diff.mean(dim=(-2, -1))

        return likelihood.transpose(0, 1)
