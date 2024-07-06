"""
@created by: heyao
@created at: 2022-08-28 16:30:33
"""
import numpy as np


def regular_mask_aug_without_pad(input_ids, tokenizer, mask_ratio=.25):
    all_inds = np.arange(1, len(input_ids) - 1)  # make sure CLS and SEP not masked
    n_mask = max(int(len(all_inds) * mask_ratio), 1)
    np.random.shuffle(all_inds)
    mask_inds = all_inds[:n_mask]
    input_ids[mask_inds] = tokenizer.mask_token_id
    return input_ids


if __name__ == '__main__':
    from transformers import AutoTokenizer

    model_path = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(tokenizer.mask_token_id)
    input_ids = np.array([1, 232, 231, 342, 3623, 3423, 5314, 2])
    print(regular_mask_aug_without_pad(input_ids, tokenizer=tokenizer))
