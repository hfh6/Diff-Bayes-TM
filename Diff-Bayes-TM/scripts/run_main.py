import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
import torch
# import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Net_params.autoencoder import TemporalAwarePRE
from Utils.random_utils import setup_seed
from Utils.metric_utils import NMAE, NRMSE
from scripts.config_helper import parse_args
from Utils.data_utils import build_dataloader
from Net_params.transformer import Transformer
from Utils.em_utils import expectation_maximization
from Diffusions.diffusion_model import GaussianDiffusion
from Diffusions.diffusion_sampler import DiffusionSampler
from TME_train_test.diffusion_trainer import DiffusionTrainer
from TME_train_test.test import monte_carlo_inference


def main(args):
    setup_seed(args.seed)
    train_loader, test_loader, scale = build_dataloader(data_root=args.data_root, dataset_name=args.dataset, flow_known_rate=args.flow_known_rate, 
                                                        batch_size=args.batch_size, random_seed=args.seed, mode=args.mode, train_size=args.train_size, 
                                                        test_size=args.test_size, window=args.seq_length)
    print("Data loaded successfully!")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # if args.flow_known_rate < 0.1:
    #     dropout_ratio = 0.5
    # else:
    #     dropout_ratio = 0.


    if args.dataset =="geant":
        if args.flow_known_rate < 0.02:
            dropout_ratio = 0.3
        else:
            dropout_ratio = 0.5


    if args.dataset =="abilene":
        if args.flow_known_rate < 0.02:
            dropout_ratio = 0.3
        elif args.flow_known_rate < 0.1:
            dropout_ratio = 0.1
        elif args.flow_known_rate < 0.5:
            dropout_ratio = 0.
        else:
            dropout_ratio = 0.7


    feature_size = train_loader.dataset.dim

    tmc_model =TemporalAwarePRE(input_size=feature_size, output_size=feature_size,ratio= dropout_ratio)
    model = Transformer(n_feat=feature_size, n_layer_enc=args.num_te, n_layer_dec=args.num_td, n_embd=args.hidden_size, 
                        n_heads=args.num_heads, attn_pdrop=args.dropout, resid_pdrop=args.dropout, max_len=args.seq_length)

    diffusion = GaussianDiffusion(model=model, seq_length=args.seq_length, timesteps=args.time_steps, 
                                  sampling_timesteps=args.sample_steps, loss_type=args.loss_type, 
                                  objective='pred_x0', beta_schedule=args.beta_schedule)

    trainer = DiffusionTrainer(tmc_model, diffusion, train_loader, results_folder=f'./CPT/Checkpoints_{args.dataset}_{args.flow_known_rate}', 
                               train_lr=args.base_lr, warmup_lr=args.warmup_lr, save_cycle=args.save_cycle, train_num_steps=args.train_epochs, 
                               gradient_accumulate_every=args.accumulate_cycle, ema_update_every=args.ema_cycle, ema_decay=args.ema_decay, 
                               patience=args.patience, min_lr=args.min_lr, threshold=args.threshold, warmup=args.warmup, factor=args.factor,current_data_rate=args.flow_known_rate)

    if args.is_train:
        trainer.train()
    else:
        trainer.load(milestone=10)

    trained_diffusion = trainer.ema.ema_model
    trained_diffusion.eval()

    sampler = DiffusionSampler(trained_diffusion)

    if args.sample_style == 'monte_carlo':
        estimations, masks, reals = monte_carlo_inference(sampler, test_loader, device=trainer.device, problem=args.mode, scale=scale)

    with torch.no_grad():
        if args.mode == 'NT':
            link_loads = test_loader.dataset.link_data.to(trainer.device)
            rm = test_loader.dataset.rm.to(trainer.device)
            estimations = expectation_maximization(torch.from_numpy(estimations).float().to(trainer.device),
                                                   link_loads, rm, num_epoch=10).cpu().numpy()

    masks = (masks > 0).reshape(masks.shape)
    nmae = NMAE(reals, estimations, masks)
    nrmse = NRMSE(reals, estimations, masks)
    print(nmae, nrmse)

if __name__ == "__main__":
    args = parse_args()
    main(args)