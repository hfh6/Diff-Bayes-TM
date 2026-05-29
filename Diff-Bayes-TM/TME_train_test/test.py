import torch
import numpy as np

from torch.utils.data import DataLoader

from Diffusions.diffusion_sampler import DiffusionSampler



def monte_carlo_inference(
    sampler: DiffusionSampler,
    dataloader: DataLoader,
    device: torch.device,
    problem: str = 'TMC',
    scale: float = 10 ** 7,
    num_particles: int = 64
):

    if problem == 'NT':
        A = dataloader.dataset.rm.to(device)

    estimations = np.empty([0, dataloader.dataset.dim])
    reals = np.empty([0, dataloader.dataset.dim])
    masks = np.empty([0, dataloader.dataset.dim])

    for idx, (x1, x2, x3) in enumerate(dataloader):

        if problem == 'TMC':
            x, m1, m2 = x1.to(device), x2.to(device), x3.to(device)
            m = m1 * (1 - m2)
            x_hat = sampler.sample_w_monte(x.shape, x*m2, partial_mask=m2, log_scale=scale, problem=problem,
                                           clip_denoised=True, initial_num_particles=num_particles)
        elif problem == 'NT':
            x, y, m = x1.to(device), x2.to(device), x3.to(device)
            x_hat = sampler.sample_w_monte(x.shape, y, A=A, log_scale=scale, problem=problem,
                                           clip_denoised=True, initial_num_particles=num_particles)


        estimations = np.row_stack([estimations, x_hat.reshape(-1, x_hat.shape[-1]).detach().cpu().numpy()])

        reals = np.row_stack([reals, x.reshape(-1, x.shape[-1]).detach().cpu().numpy()])

        masks = np.row_stack([masks, m.reshape(-1, m.shape[-1]).detach().cpu().numpy()])

        torch.cuda.empty_cache()

    return estimations, masks, reals
