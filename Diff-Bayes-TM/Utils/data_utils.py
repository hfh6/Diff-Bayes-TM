import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Utils.random_utils import random_mask, link_select





def build_dataloader(
        data_root: str,
        dataset_name: str,
        flow_known_rate: float,
        batch_size: int,
        random_seed: int,
        link_known_rate: float = 1.,
        train_size: int = 3000,
        test_size: int = 672,
        window: int = 12,
        mode: str = 'NT'
):

    if dataset_name == 'abilene':
        scale = 10 ** 9
        tm_filename = 'abilene_tm.csv'
        rm_filename = 'abilene_rm.csv'
    elif dataset_name == 'geant':
        scale = 10 ** 7
        tm_filename = 'geant_tm.csv'
        rm_filename = 'geant_rm.csv'
    elif dataset_name == 'cernet':
        scale = 10 ** 7
        tm_filename = 'cernet_tm.csv'
        rm_filename = 'cernet_rm.csv'


    current_dir = os.path.dirname(os.path.abspath(__file__))

    tm_filepath = os.path.join(current_dir, "..", "Datasets", tm_filename)

    tm_filepath = os.path.normpath(tm_filepath)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    rm_filepath = os.path.join(current_dir, "..", "Datasets", rm_filename)

    rm_filepath = os.path.normpath(rm_filepath)

    print(f"尝试访问的TM文件路径: {os.path.abspath(tm_filepath)}")
    print(f"尝试访问的RM文件路径: {os.path.abspath(rm_filepath)}")

    train_dataset = TMDataset(tm_filepath, train_size, 0, flow_known_rate, window, scale, random_seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if mode == 'NT':
        test_dataset = NTDataset(tm_filepath, rm_filepath, test_size, train_size, link_known_rate,
                                 window, scale, random_seed)
    elif mode == 'TMC':
        test_dataset = TMCDataset(tm_filepath, test_size, 0, flow_known_rate, window, scale, random_seed)
    else:
        raise ValueError(f"Unknown mode: {mode}. Please choose 'NT' or 'TMC'.")

    test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, num_workers=0)

    return train_loader, test_loader, scale


class TMDataset(Dataset):


    def __init__(
            self,
            tm_filepath: str,
            num_samples: int,
            start_time: int = None,
            known_rate: float = 0.1,
            window: int = 12,
            scale: float = 10 ** 9,
            seed: int = 2025
    ):
        super(TMDataset, self).__init__()
        data = self.read_data(tm_filepath, num_samples, start_time, scale)
        tms, _, mask = random_mask(data, 1 - known_rate, seed)

        self.window = window
        len, self.dim = tms.shape
        self.sample_num = max(len - self.window + 1, 0)

        self.samples, self.masks = self.get_samples(tms, mask)

    def read_data(
            self,
            tm_filepath: str,
            train_size: int,
            start_time: int,
            scale: float
    ):

        if not os.path.exists(tm_filepath):
            raise FileNotFoundError(f"File {tm_filepath} does not exist.")
        if start_time is None:
            start_time = 0


        df = pd.read_csv(tm_filepath, header=None)
        df.drop(df.columns[-1], axis=1, inplace=True)


        traffic = df.values[start_time:(train_size + start_time)] / scale


        quantile = np.percentile(df.values / scale, q=99)
        traffic = np.clip(traffic, 0, quantile) / quantile

        return traffic

    def get_samples(self, data: torch.Tensor, mask: torch.Tensor):

        x = torch.zeros((self.sample_num, self.window, self.dim))
        m = torch.zeros((self.sample_num, self.window, self.dim))

        for i in range(self.sample_num):
            start = i
            end = i + self.window
            x[i, :, :] = data[start:end, :]
            m[i, :, :] = mask[start:end, :]

        return x, m

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        x = self.samples[idx, :, :]
        m = self.masks[idx, :, :]
        return x, m

    def update(self, updated_tms: torch.Tensor):

        self.samples = updated_tms


class TMCDataset(Dataset):


    def __init__(
            self,
            tm_filepath: str,
            num_samples: int,
            start_time: int = None,
            known_rate: float = 0.1,
            window: int = 12,
            scale: float = 10 ** 9,
            seed: int = 2025
    ):
        super(TMCDataset, self).__init__()
        data = self.read_data(tm_filepath, num_samples, start_time, scale)
        self.tms, self.mask_1, self.mask_2 = random_mask(data, 1 - known_rate, seed)

        self.window = window
        _, self.dim = self.tms.shape

        assert data.shape[0] % self.window == 0, '!! The number of samples should be divisible by the window size.'
        self.sample_num = int(data.shape[0] // self.window)

        self.samples, self.masks_ob, self.masks = self.get_samples(self.tms, self.mask_1, self.mask_2)

    def read_data(
            self,
            tm_filepath: str,
            train_size: int,
            start_time: int,
            scale: float
    ):

        if not os.path.exists(tm_filepath):
            raise FileNotFoundError(f"File {tm_filepath} does not exist.")
        if start_time is None:
            start_time = 0


        df = pd.read_csv(tm_filepath, header=None)
        df.drop(df.columns[-1], axis=1, inplace=True)


        traffic = df.values[start_time:(train_size + start_time)] / scale


        quantile = np.percentile(df.values / scale, q=99)
        traffic = np.clip(traffic, 0, quantile) / quantile

        return traffic

    def get_samples(self, data: torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor):

        x = torch.zeros((self.sample_num, self.window, self.dim))
        m1 = torch.zeros((self.sample_num, self.window, self.dim))
        m2 = torch.zeros((self.sample_num, self.window, self.dim))

        j = 0
        for i in range(0, self.sample_num):
            start = j
            end = j + self.window
            x[i, :, :] = data[start:end, :]
            m1[i, :, :] = mask1[start:end, :]
            m2[i, :, :] = mask2[start:end, :]
            j = end

        return x, m1, m2

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        x = self.samples[idx, :, :]
        m1 = self.masks_ob[idx, :, :]
        m2 = self.masks[idx, :, :]
        return x, m1, m2

    def extend_2_tmc(self):

        pass


class NTDataset(Dataset):


    def __init__(
            self,
            tm_filepath: str,
            rm_filepath: str,
            num_samples: int,
            start_time: int = None,
            link_known_rate: float = 1.,
            window: int = 12,
            scale: float = 10 ** 9,
            seed: int = 2025
    ):
        super(NTDataset, self).__init__()
        self.window = window

        data, rm = self.read_data(tm_filepath, rm_filepath, num_samples, start_time, scale)
        link_data = data @ rm

        self.dim, self.link_dim = rm.shape
        self.rm = torch.from_numpy(rm).transpose(0, 1).float()

        tms, mask_ob, _ = random_mask(data, 0., seed)



        self.link_data = torch.from_numpy(link_data).float()

        assert data.shape[0] % self.window == 0, '!! The number of samples should be divisible by the window size.'
        self.sample_num = int(data.shape[0] // self.window)

        self.tm_samples, self.link_samples, self.masks = self.get_samples(tms, self.link_data, mask_ob)



    def read_data(
            self,
            tm_filepath: str,
            rm_filepath: str,
            test_size: int,
            start_time: int,
            scale: float
    ):

        if not os.path.exists(tm_filepath):
            raise FileNotFoundError(f"File {tm_filepath} does not exist.")
        if not os.path.exists(rm_filepath):
            raise FileNotFoundError(f"File {rm_filepath} does not exist.")
        if start_time is None:
            start_time = 0


        df = pd.read_csv(tm_filepath, header=None)
        df.drop(df.columns[-1], axis=1, inplace=True)


        rm_df = pd.read_csv(rm_filepath, header=None)
        rm_df.drop(rm_df.columns[-1], axis=1, inplace=True)


        traffic = df.values[start_time:(test_size + start_time)] / scale


        quantile = np.percentile(df.values / scale, q=99)
        traffic = np.clip(traffic, 0, quantile) / quantile

        return traffic, rm_df.values

    def get_samples(self, data1: torch.Tensor, data2: torch.Tensor, mask: torch.Tensor):

        x = torch.zeros((self.sample_num, self.window, self.dim))
        y = torch.zeros((self.sample_num, self.window, self.link_dim))
        m = torch.zeros((self.sample_num, self.window, self.dim))

        j = 0
        for i in range(0, self.sample_num):
            start = j
            end = j + self.window
            x[i, :, :] = data1[start:end, :]
            y[i, :, :] = data2[start:end, :]
            m[i, :, :] = mask[start:end, :]
            j = end

        return x, y, m

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        x = self.tm_samples[idx, :, :]
        y = self.link_samples[idx, :, :]
        m = self.masks[idx, :, :]
        return x, y, m

    @staticmethod
    def padding_sqare(mat: torch.Tensor):

        zeros = torch.zeros((mat.shape[1] - mat.shape[0], mat.shape[1]), device=mat.device)
        square_mat = torch.concatenate((mat, zeros), axis=0)
        return square_mat




if __name__ == "__main__":
    pass