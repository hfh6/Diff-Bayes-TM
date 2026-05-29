import torch
import random
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def link_select(num: int, missing_ratio: float = 0.5, seed: int = 1999):

    unknown_num = int(np.ceil(num * missing_ratio))


    st0 = np.random.get_state()
    np.random.seed(seed)

    id_rdm = torch.randperm(num).cpu().numpy()
    ex_fs = id_rdm[:unknown_num]


    np.random.set_state(st0)
    return ex_fs


def random_mask(
    observed_values: np.ndarray, 
    missing_ratio: float = 0., 
    seed: int = 1999, 
    exclude_features: int = None, 
    exclude_zeros: bool = True
):

    observed_masks = ~np.isnan(observed_values)


    if exclude_zeros:
        observed_masks[observed_values==0.] = False
    
    if exclude_features is not None:
        observed_masks[:, exclude_features] = False


    masks = observed_masks.reshape(-1).copy()
    obs_indices = np.where(masks)[0].tolist()


    st0 = np.random.get_state()
    np.random.seed(seed)

    miss_indices = np.random.choice(
        obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
    )


    np.random.set_state(st0)
    
    masks[miss_indices] = False
    gt_masks = masks.reshape(observed_masks.shape)

    observed_values = np.nan_to_num(observed_values)
    return torch.from_numpy(observed_values).float(), torch.from_numpy(observed_masks).float(),\
           torch.from_numpy(gt_masks).float()