import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('请输入布尔值（True/False）')


def parse_args(notebook=False):
    parser = argparse.ArgumentParser(description="Toy Experiment", allow_abbrev=False)

    # args for random

    parser.add_argument('--seed', type=int, default=12345,
                        help='seed for initializing training')
    parser.add_argument('--data_seed', type=int, default=123,
                        help='seed for loading data')

    # args for dataset

    parser.add_argument('--data_root', type=str, default="./Datasets",
                        help="Root Dir of .csv File")
    parser.add_argument('--dataset', type=str, default="geant",
                        choices=["abilene", "geant"],
                        help="Dataset Name")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size")
    parser.add_argument('--flow_known_rate', type=float, default=0.02,
                        help="Known Ratio (TMs during Training and Testing)")

    parser.add_argument('--mode', type=str, default="TMC",
                        choices=['NT', 'TMC'],
                        help="Type of TM-related Tasks")
    parser.add_argument('--train_size', type=int, default=3000, help='Number of Training Samples')
    parser.add_argument('--test_size', type=int, default=672, help='Number of Testing Samples')
    parser.add_argument('--seq_length', type=int, default=12, help='Length of Processed Sequence')



    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden Size of MLP layers')
    parser.add_argument('--num_te', type=int, default=2, help='Number of Transformer Encoder layers')
    parser.add_argument('--num_td', type=int, default=2, help='Number of Transformer Decoder layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of Attention heads')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout Rate in Transformer')

    parser.add_argument('--self_condition', type=bool, default=True,
                        help='Use Self-Condition or not.')
    parser.add_argument('--time_steps', type=int, default=300,
                        help='Number of Diffusion Steps.')
    parser.add_argument('--sample_steps', type=int, default=300,
                        help='Number of Sampling Steps.')
    parser.add_argument('--loss_type', type=str, default='l1',
                        choices=['l1', 'l2'], help='Type of Loss Function.')
    parser.add_argument('--beta_schedule', type=str, default='cosine',
                        choices=['linear', 'cosine'],
                        help='Type of Beta Schedule.')



    parser.add_argument('--base_lr', type=float, default=1e-5,
                        help='Learning Rate before Warmup.')
    parser.add_argument('--warmup_lr', type=float, default=8e-4,
                        help='Learning Rate after Warmup.')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='Minimum Learning Rate.')
    parser.add_argument('--warmup', type=int, default=500,
                        help='Number of Warmup Epochs.')
    parser.add_argument('--patience', type=int, default=2000,
                        help='Patience.')
    parser.add_argument('--threshold', type=float, default=1e-1,
                        help='Hyperparameter for Evaluating whether Better or not.')
    parser.add_argument('--factor', type=float, default=0.5,
                        help='Hyperparameter for Reducing Learning Rate.')
    parser.add_argument('--ema_cycle', type=int, default=10,
                        help='Number of Epochs between Two EMA Updating.')
    parser.add_argument('--ema_decay', type=float, default=0.995,
                        help='Decay Rate of EMA.')
    parser.add_argument('--train_epochs', type=int, default=10000,
                        help='Number of Training Epochs.')
    parser.add_argument('--save_cycle', type=int, default=1000,
                        help='Number of Epochs between Two Model Saving.')
    parser.add_argument('--accumulate_cycle', type=int, default=2,
                        help='Number of Epochs between Two Gradient Descent.')

    parser.add_argument('--is_train', type=str2bool, nargs='?', const=True, default=False,
                        help='训练模式设为True，推理模式设为False。')


    parser.add_argument('--sample_style', type=str, default="monte_carlo",
                        choices=[ "monte_carlo"],
                        help="Dataset Name")
    parser.add_argument('--guide_rate', type=float, default=5e-2,
                        help="Learning Rate for Guidance during Inference")

    if notebook:
        args = parser.parse_known_args()[0]
    else:
        args = parser.parse_args()

    return args
