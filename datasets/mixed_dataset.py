"""
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np

from .base_dataset import BaseDataset

# class MixedDataset(torch.utils.data.Dataset):

#     def __init__(self, options, **kwargs):
#         # self.dataset_list = ['h36m', 'lsp-orig', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp']
#         self.dataset_list = ['lsp-orig', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp']
#         self.dataset_list = ['coco']
#         # self.dataset_dict = {'h36m': 0, 'lsp-orig': 1, 'mpii': 2, 'lspet': 3, 'coco': 4, 'mpi-inf-3dhp': 5}
#         self.dataset_dict = { 'coco': 0}
#         self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
#         total_length = sum([len(ds) for ds in self.datasets])
#         length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
#         self.length = max([len(ds) for ds in self.datasets])
#         """
#         Data distribution inside each batch:
#         30% H36M - 60% ITW - 10% MPI-INF
#         """
#         self.partition = [len(self.datasets[0])]
#         self.partition = np.array(self.partition).cumsum()

#     def __getitem__(self, index):
#         p = np.random.rand()
#         for i in range(6):
#             if p <= self.partition[i]:
#                 return self.datasets[i][index % len(self.datasets[i])]

#     def __len__(self):
#         return self.length

class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, options, **kwargs):
        # Restricting to only COCO dataset
        self.dataset_list = ['coco']
        self.dataset_dict = {'coco': 0}
        
        # Loading only COCO dataset
        self.datasets = [BaseDataset(options, 'coco', **kwargs)]
        
        # Set the total length to the length of the COCO dataset
        self.length = len(self.datasets[0])

    def __getitem__(self, index):
        # Directly fetch from the single dataset (COCO)
        return self.datasets[0][index % self.length]

    def __len__(self):
        return self.length