from typing import List

import torch

from soccer_eventpred.data.dataclass import Instance


class SoccerEventDataset(torch.utils.data.Dataset):
    def __init__(self):
        self._dataset: List[Instance] = []

    def add(self, instance):
        self._dataset.append(instance)

    def __getitem__(self, idx):
        return self._dataset[idx]

    def __len__(self):
        return len(self._dataset)
