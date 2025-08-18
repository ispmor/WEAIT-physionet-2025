import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import random
from itertools import cycle

class GuaranteePositiveBatchSampler(Sampler):
    def __init__(self, labels, batch_size, min_positives=10, drop_last=True, seed=None):
        self.labels = list(labels)                   # 0/1 per sample (len == len(dataset))
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = random.Random(seed)
        self.min_positives = min_positives

        self.pos_idx = [i for i, y in enumerate(self.labels) if y == 1]
        self.neg_idx = [i for i, y in enumerate(self.labels) if y == 0]
        if not self.pos_idx:
            raise ValueError("No positive samples in dataset.")

    def __len__(self):
        n = len(self.labels) // self.batch_size
        return n if self.drop_last else (len(self.labels) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # shuffle each epoch
        self.rng.shuffle(self.pos_idx)
        self.rng.shuffle(self.neg_idx)

        pos_iter = cycle(self.pos_idx)                          # reuse positives if needed
        neg_iter = iter(self.neg_idx)

        batches = []
        batch = []

        # number of batches to yield this epoch
        for _ in range(len(self)):
            # ensure min_positive positive
            
            batch = [next(pos_iter) for i in range(self.min_positives)]
            # fill the rest with negatives (wrap to negatives->positives if we run out)
            for _ in range(self.batch_size - self.min_positives):
                try:
                    idx = next(neg_iter)
                except StopIteration:
                    # if negatives exhausted, use positives to fill
                    idx = next(pos_iter)
                batch.append(idx)
            self.rng.shuffle(batch)
            batches.append(batch)

        return iter(batches)
