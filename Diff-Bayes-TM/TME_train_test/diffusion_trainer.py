import os
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.nn.utils import clip_grad_norm_
from TME_train_test.lr_schedule import ReduceLROnPlateauWithWarmup


def cycle(dl):
    while True:
        for data in dl:
            yield data


class DiffusionTrainer(object):
    def __init__(
        self,
        tmc_model,
        model, 
        data_loader, 
        results_folder='./Checkpoints', 
        train_lr=1e-5, 
        warmup_lr=1e-4, 
        save_cycle=10000,
        train_num_steps=100000, 
        adam_betas=(0.9, 0.96), 
        gradient_accumulate_every=2, 
        ema_update_every=10,
        ema_decay=0.995,
        patience=1000, 
        min_lr=1e-6, 
        threshold=0., 
        warmup=2000,
        factor=0.5,
        #新增
        current_data_rate=0.02
    ):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.tmc_model = tmc_model.to(self.device)


        self.pre_epoch = 10000
        self.train_num_steps = train_num_steps

        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_cycle = int(save_cycle)
        self.dataloader = data_loader
        self.dl = cycle(data_loader)
        self.step = 0
        self.milestone = 0
        self.current_data_rate=current_data_rate
        self.results_folder = Path(results_folder)
        os.makedirs(self.results_folder, exist_ok=True)

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=train_lr, betas=adam_betas)
        self.sch = ReduceLROnPlateauWithWarmup(optimizer=self.opt, factor=factor, patience=patience, min_lr=min_lr, threshold=threshold,
                                               threshold_mode='rel', warmup_lr=warmup_lr, warmup=warmup, verbose=False)
        

        self.criteon = nn.L1Loss().to(self.device)
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        self.tmc_opt = Adam(filter(lambda p: p.requires_grad, self.tmc_model.parameters()), lr=1e-3, betas=adam_betas)

    def save(self, milestone):
        if milestone < 10:
            return
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'tmc_model': self.tmc_model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    def load(self, milestone):
        device = self.device
        self.milestone = milestone
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.model.load_state_dict(data['model'])
        self.tmc_model.load_state_dict(data['tmc_model'])

    def train(self):
        device = self.device
        step = 0
        
        print("Start Pre-processing Network Training.")

        with tqdm(initial=step, total=10000) as pbar:

            while step < 10000:

                x, m = next(self.dl)
                x, m = x.to(device), m.to(device)



                x_hat, features = self.tmc_model(x * m, return_features=True)


                recon_loss = self.criteon(x_hat * m, x * m)

                contrastive_loss = self.tmc_model.compute_contrastive_loss(features)

                total_loss = recon_loss + self.tmc_model.contrastive_weight * contrastive_loss


                total_loss.backward()

                pbar.set_description(
                    f'recon_loss: {recon_loss.item():.6f}, contra_loss: {contrastive_loss.item():.6f}, total_loss: {total_loss.item():.6f}')
                self.tmc_opt.step()
                self.tmc_opt.zero_grad()

                step += 1
                pbar.update(1)

        self.tmc_model.requires_grad_(False)
        self.tmc_model.eval()

        print("Finish Pre-processing Network Training.")

        restoration = np.empty([0, self.dataloader.dataset.window, self.dataloader.dataset.dim])
        for idx, (x, m) in enumerate(self.dataloader):
            x, m = x.to(self.device), m.to(self.device)

            x_hat = self.tmc_model.preprocessing(x * m, x * m, m)

            restoration = np.row_stack([restoration, x_hat.detach().cpu().numpy()])
        restoration = torch.from_numpy(restoration).float()

        print("Now Start Diffusion Model Training.")

        self.dataloader.dataset.update(restoration)

        self.dl_update = cycle(self.dataloader)
        step = 0

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    x, m = next(self.dl_update)
                    x, m = x.to(device), m.to(device)

                    x_hat = self.model(x)

                    loss = self.criteon(x_hat * m, x * m)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                step += 1
                self.step += 1
                self.ema.update()

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)

                pbar.update(1)

        print('Training Complete')

