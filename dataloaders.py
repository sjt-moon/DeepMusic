import itertools
import copy
import numpy as np


class Dataloader:
    MIN_SEQ_LEN = 2

    def __init__(self, dataset, batch_size=1, strict_len_lims=False):
        """
        :param dataset: list of training sequences
        :param batch_size: the batch size
        :param strict_len_lims: whether to strictly apply the `len_lims` at the
               end of the dataset
        """
        self.dataset = dataset
        self.strict_len_lims = strict_len_lims
        self.batch_size = batch_size

    @staticmethod
    def sort_seqs(seqs):
        seqs.sort(key=len, reverse=True)


class AcrossTuneDataloader(Dataloader):
    def __init__(self, dataset, len_lims, batch_size=1, strict_len_lims=False):
        """
        :param len_lims: (lower limit, 1 + upper limit)
        """
        Dataloader.__init__(self, dataset, batch_size=batch_size,
                            strict_len_lims=strict_len_lims)
        self.len_lims = len_lims
        assert len_lims[0] < len_lims[1]
        self.tune_char_cursor = None
        self.iterating_dataset = None
        self.renew_iterating_dataset()

    def renew_iterating_dataset(self):
        perm = np.random.permutation(len(self.dataset))
        self.iterating_dataset = list(itertools.chain.from_iterable(
            self.dataset[i] for i in perm))
        self.tune_char_cursor = 0
        assert self.len_lims[1] < len(self.iterating_dataset)


class AcrossTuneNonOverlapDataloader(AcrossTuneDataloader):
    def __init__(self, dataset, len_lims, batch_size=1, strict_len_lims=False):
        AcrossTuneDataloader.__init__(self, dataset, len_lims,
                                      batch_size=batch_size,
                                      strict_len_lims=strict_len_lims)

    def next(self):
        seqs = []
        for i in range(self.batch_size):
            l = np.random.randint(*self.len_lims)
            if self.tune_char_cursor + l > len(self.iterating_dataset):
                if self.strict_len_lims:
                    self.renew_iterating_dataset()
                    l = np.random.randint(*self.len_lims)
                else:
                    l = len(self.iterating_dataset) - self.tune_char_cursor
                    if l < Dataloader.MIN_SEQ_LEN:
                        self.renew_iterating_dataset()
                        l = np.random.randint(*self.len_lims)
            seqs.append(self.iterating_dataset[
                        self.tune_char_cursor:self.tune_char_cursor + l])
            self.tune_char_cursor += l
        self.sort_seqs(seqs)
        return seqs


class AcrossTuneRandomDataloader(AcrossTuneDataloader):
    def __init__(self, dataset, len_lims, batch_size=1, strict_len_lims=False):
        AcrossTuneDataloader.__init__(self, dataset, len_lims,
                                      batch_size=batch_size,
                                      strict_len_lims=strict_len_lims)

    def next(self):
        seqs = []
        for i in range(self.batch_size):
            s = np.random.randint(len(self.iterating_dataset)
                                  - Dataloader.MIN_SEQ_LEN)
            l = np.random.randint(*self.len_lims)
            if s + l > len(self.iterating_dataset):
                if not self.strict_len_lims:
                    l = len(self.iterating_dataset) - s
                    assert l >= Dataloader.MIN_SEQ_LEN
                else:
                    while s + l > len(self.iterating_dataset):
                        s = np.random.randint(len(self.iterating_dataset)
                                              - Dataloader.MIN_SEQ_LEN)
                        l = np.random.randint(*self.len_lims)
            seqs.append(self.iterating_dataset[s:s + l])
        self.sort_seqs(seqs)
        return seqs


class ValidationDataloader(Dataloader):
    def __init__(self, dataset):
        Dataloader.__init__(self, dataset)

    def next(self):
        seqs = copy.copy(self.dataset)
        self.sort_seqs(seqs)
        return seqs
