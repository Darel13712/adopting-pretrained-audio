import pandas as pd
import numpy as np
import torch
import scipy
from torch.utils.data import Dataset, DataLoader


class InteractionDataset(Dataset):
    def __init__(self, df, neg_samples=20, device=None):
        """One positive vs 20 negatives"""
        self.df = df
        self.neg_samples = neg_samples

        self.user_history = df.groupby('user_id')['item_id'].agg(list).to_dict()
        all_items = set(df.item_id.unique())
        self.user_negatives = {user: list(all_items - set(self.user_history[user])) for user in self.user_history}
        self.device = torch.device(device or "cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        user = torch.tensor(sample.user_id, dtype=torch.long, device=self.device)
        positive_item = torch.tensor(sample.item_id, dtype=torch.long, device=self.device)
        negative_items = self._sample_negatives(sample.user_id)
        return user, positive_item, negative_items

    def _sample_negatives(self, user):
        negatives = self.user_negatives[user]
        res = np.random.choice(negatives, self.neg_samples)
        return torch.tensor(res, dtype=torch.long, device=self.device)

# todo: write item sampling and training version
class InteractionDatasetItems(Dataset):
    """All positive users and same number of negative users"""
    def __init__(self, df, neg_samples=20):
        self.df = df
        self.neg_samples = neg_samples
        #
        # self.item_history = df.groupby('item_id')['user_id'].agg(list).to_dict()
        # all_users = set(df.user_id.unique())
        # self.item_negatives = {item: list(all_users - set(self.item_history[item])) for item in self.item_history}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        item = torch.tensor(sample.item_id, dtype=torch.long)
        positive_user = torch.tensor(sample.user_id, dtype=torch.long)
        # negative_users = self._sample_negatives(sample.item_id)
        confidence = torch.tensor(self._confidence(sample['count']), dtype=torch.long)
        # return item, positive_user, negative_users, confidence
        return item, positive_user, confidence

    def _sample_negatives(self, item):
        negatives = self.item_negatives[item]
        res = np.random.choice(negatives, self.neg_samples)
        return torch.tensor(res, dtype=torch.long)

    def _confidence(self, x):
        return x

class RecommenderDataset(Dataset):
    def __init__(self, dataframe):
        self.users = torch.tensor(dataframe['user_id'].values)
        self.items = torch.tensor(dataframe['item_id'].values)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

class ContrastiveDataset(Dataset):
    def __init__(self, df, device=None):
        unique_items = df.item_id.unique()
        self.items = unique_items
        self.device = torch.device(device or "cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = torch.tensor(self.items[idx], dtype=torch.long, device=self.device)
        return item

class SparseMatrixDataset(Dataset):
    def __init__(self, sparse_matrix: scipy.sparse.csr_matrix, device=None):
        self.csr_matrix = sparse_matrix
        self.device = torch.device(device or "cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return self.csr_matrix.shape[0]

    def __getitem__(self, idx):
        """
        Extract a row of a sparse matrix converted to sparse coo tensor allocated on the CPU.
        To same memory bandwidth, it moves data in sparse format to given device (preferably GPU) and convert it to dense there
        """
        scipy_coo = self.csr_matrix[idx].tocoo()
        torch_coo = torch.sparse_coo_tensor(
            np.vstack([scipy_coo.row, scipy_coo.col]),
            scipy_coo.data.astype(np.float32),
            scipy_coo.shape,
        )
        return torch_coo.to(self.device).to_dense()


