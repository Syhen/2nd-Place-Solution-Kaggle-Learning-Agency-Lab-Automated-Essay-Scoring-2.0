"""
@created by: heyao
@created at: 2022-08-25 01:24:33
"""
import math

import torch
import transformers


class SequenceBucketPadCollator(object):
    def __init__(self, max_length, tokenizer: transformers.PreTrainedTokenizer):
        self.max_length = max_length
        self.tokenizer = tokenizer

    def _get_max_length(self, x, key="attention_mask"):
        max_len = max([sum(i[key]) for i in x])
        max_len = min(self.max_length, max_len)
        return max_len

    def __call__(self, batch):
        """

        :param batch:
        :return:
        """
        x = batch
        xs = {}
        max_len = self._get_max_length(x)
        if "longformer" in self.tokenizer.name_or_path:
            max_len = int(512 * math.ceil(max_len / 512))
        if "rel_input_ids" in x[0].keys():
            max_len2 = self._get_max_length(x, "rel_attention_mask")
        y = None
        y_aux = None
        for key in x[0].keys():
            if "labels" in key:
                pad_val = -100
            else:
                pad_val = self.tokenizer.pad_token_id
            if key not in ["labels", "mask_token_indexes", "is_pc2", "aux_labels", "rel_labels", "ref_labels",
                           "soft_labels"]:
                # print(max_len)
                # print(key)
                # print([len(i[key]) for i in x])
                if "rel" in key:
                    xs[key] = torch.vstack([torch.LongTensor(i[key] + [pad_val] * (max_len2 - len(i[key]))) for i in x])
                else:
                    xs[key] = torch.vstack([torch.LongTensor(i[key] + [pad_val] * (max_len - len(i[key]))) for i in x])
            elif key in ["mask_token_indexes", "is_pc2"]:
                xs[key] = torch.LongTensor([i[key] for i in x])
            elif key in ["soft_labels", "rel_labels", "ref_labels"]:
                xs[key] = torch.FloatTensor([i[key] for i in x])
            elif key == "aux_labels":
                y_aux = torch.FloatTensor([i[key] for i in x])
            else:
                y = torch.FloatTensor([i[key] for i in x])
        if y is not None:
            if y_aux is not None:
                return xs, y, y_aux
            return xs, y
        return xs


class MaxSeqLenPadCollator(SequenceBucketPadCollator):
    def __init__(self, max_length, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__(max_length, tokenizer)

    def _ensure_max_length(self, x):
        return self.max_length


if __name__ == '__main__':
    from transformers import AutoTokenizer
    from aes2.utils.dataset.simple import CompetitionDataset

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    collate_fn = SequenceBucketPadCollator(max_length=1024, tokenizer=tokenizer)
    data = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": 1},
        {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1], "labels": 2},
        {"input_ids": [1, 2, 3, 5], "attention_mask": [1, 1, 1, 1], "labels": 3},
        {"input_ids": [1, 2, 2, 5], "attention_mask": [1, 1, 1, 1], "labels": 4}
    ]
    dataset = CompetitionDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
    it = iter(dataloader)
    while 1:
        try:
            print(next(it))
        except StopIteration:
            print(next(dataloader.__iter__()))
            break
