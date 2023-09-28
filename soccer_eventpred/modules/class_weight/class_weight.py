from typing import List, Optional

import torch
from tango.common import Registrable

from soccer_eventpred.modules.datamodule.soccer_dataset import SoccerEventDataset


class ClassWeightBase(Registrable):
    def calculate(
        self,
        dataset: SoccerEventDataset,
        num_classes: int,
        ignore_indices: Optional[List[int]] = None,
        class_counts: Optional[List[int]] = None,
    ):
        raise NotImplementedError


@ClassWeightBase.register("reciprocal")
class ClassWeightReciprocal(ClassWeightBase):
    def calculate(
        self,
        dataset: SoccerEventDataset,
        num_classes: int,
        ignore_indices: Optional[List[int]] = None,
        class_counts: Optional[List[int]] = None,
    ):
        if class_counts is None:
            class_weights = torch.ones(num_classes)
            for i in range(len(dataset)):
                instance = dataset[i]
                for event in instance.event_ids:
                    class_weights[event] += 1
        else:
            class_weights = torch.tensor(class_counts, dtype=torch.float) + 1
        for class_idx in range(len(class_weights)):
            class_weights[class_idx] = 1 / class_weights[class_idx]

        # ignore_indices
        if ignore_indices is not None:
            for ignore_idx in ignore_indices:
                class_weights[ignore_idx] = 0

        return class_weights


@ClassWeightBase.register("sklearn")
class ClassWeightSklearn(ClassWeightBase):
    def calculate(
        self,
        dataset: SoccerEventDataset,
        num_classes: int,
        ignore_indices: Optional[List[int]] = None,
        class_counts: Optional[List[int]] = None,
    ):
        if class_counts is None:
            class_weights = torch.ones(num_classes)
            event_len = 0
            for i in range(len(dataset)):
                instance = dataset[i]
                for event in instance.event_ids:
                    class_weights[event] += 1
                    event_len += 1
        else:
            class_weights = torch.tensor(class_counts, dtype=torch.float) + 1
            event_len = class_weights.sum().item()
        # cf: https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
        for class_idx in range(len(class_weights)):
            class_weights[class_idx] = event_len / (
                class_weights[class_idx] * num_classes
            )
        # ignore_indices
        if ignore_indices is not None:
            for ignore_idx in ignore_indices:
                class_weights[ignore_idx] = 0
        return class_weights


@ClassWeightBase.register("exponential")
class ClassWeightExponential(ClassWeightBase):
    def __init__(self, beta: float = 0.0):
        self.beta = beta

    def calculate(
        self,
        dataset: SoccerEventDataset,
        num_classes: int,
        ignore_indices: Optional[List[int]] = None,
        class_counts: Optional[List[int]] = None,
    ):
        if class_counts is None:
            class_weights = torch.ones(num_classes)
            for i in range(len(dataset)):
                instance = dataset[i]
                for event in instance.event_ids:
                    class_weights[event] += 1
        else:
            class_weights = torch.tensor(class_counts, dtype=torch.float) + 1
        sum_exponentials = ((1 / class_weights) ** self.beta).sum().item()
        for class_idx in range(len(class_weights)):
            class_weights[class_idx] = (
                (1 / class_weights[class_idx]) ** self.beta
            ) / sum_exponentials

        # ignore_indices
        if ignore_indices is not None:
            for ignore_idx in ignore_indices:
                class_weights[ignore_idx] = 0
        return class_weights
