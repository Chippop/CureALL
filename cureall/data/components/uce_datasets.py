import torch
from unicore.data import BaseWrapperDataset


class MultiDatasetSentenceCollator(object):
    def __init__(self, pad_length):
        self.pad_length = pad_length

    def __call__(self, batch):
        batch_size = len(batch)
        batch_sentences = torch.zeros((batch_size, self.pad_length))
        mask = torch.zeros((batch_size, self.pad_length))
        cell_sentences = torch.zeros((batch_size, self.pad_length))

        idxs = torch.zeros(batch_size)

        i = 0
        max_len = 0
        for bs, msk, idx, seq_len, cs in batch:
            batch_sentences[i, :] = bs
            cell_sentences[i, :] = cs
            max_len = max(max_len, seq_len)
            mask[i, :] = msk
            idxs[i] = idx

            i += 1

        return batch_sentences[:, :max_len], mask[:, :max_len]


class UCEBaseDataset(BaseWrapperDataset):
    def __init__(self, dataset, uce_padding_length: int = 1536):
        self.dataset = dataset
        self._collater = MultiDatasetSentenceCollator(uce_padding_length)

    def __getitem__(self, index):
        try:
            return self.dataset[index]["cell_batch"]
        except:
            import ipdb; ipdb.set_trace()

    def __len__(self):
        return len(self.dataset)

    def collater(self, sample):
        return self._collater(sample)
