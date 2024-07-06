"""
@created by: heyao
@created at: 2024-06-05 00:55:24
"""
from copy import deepcopy
from dataclasses import dataclass

import torch
from transformers import DataCollatorWithPadding


@dataclass
class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    """
    data collector for seq classification
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"

    def __call__(self, features):
        span_head_idxs = [feature["span_head_idxs"] for feature in features]
        span_tail_idxs = [feature["span_tail_idxs"] for feature in features]
        span_attention_mask = [[1] * len(feature["span_head_idxs"]) for feature in features]

        sentence_head_idxs = [feature["sentence_head_idxs"] for feature in features]
        sentence_tail_idxs = [feature["sentence_tail_idxs"] for feature in features]
        sentence_attention_mask = [[1] * len(feature["span_head_idxs"]) for feature in features]

        labels = None
        if "labels" in features[0].keys():
            labels = [feature["labels"] for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        if labels is not None:
            batch["labels"] = labels

        # span related fields
        b_max = max([len(l) for l in span_head_idxs])
        sentence_b_max = max([len(l) for l in sentence_head_idxs])
        max_len = len(batch["input_ids"][0])

        default_head_idx = max(max_len - 4, 1)  # for padding
        default_tail_idx = max(max_len - 2, 1)  # for padding

        batch["span_head_idxs"] = [
            ex_span_head_idxs + [default_head_idx] * (b_max - len(ex_span_head_idxs)) for ex_span_head_idxs in
            span_head_idxs
        ]

        batch["span_tail_idxs"] = [
            ex_span_tail_idxs + [default_tail_idx] * (b_max - len(ex_span_tail_idxs)) for ex_span_tail_idxs in
            span_tail_idxs
        ]

        batch["span_attention_mask"] = [
            ex_discourse_masks + [0] * (b_max - len(ex_discourse_masks)) for ex_discourse_masks in span_attention_mask
        ]
        # ----
        batch["sentence_head_idxs"] = [
            ex_sentence_head_idxs + [default_head_idx] * (sentence_b_max - len(ex_sentence_head_idxs)) for
            ex_sentence_head_idxs in sentence_head_idxs
        ]

        batch["sentence_tail_idxs"] = [
            ex_sentence_tail_idxs + [default_tail_idx] * (sentence_b_max - len(ex_sentence_tail_idxs)) for
            ex_sentence_tail_idxs in sentence_tail_idxs
        ]

        batch["sentence_attention_mask"] = [
            ex_sentence_masks + [0] * (sentence_b_max - len(ex_sentence_masks)) for ex_sentence_masks in
            sentence_attention_mask
        ]

        batch = {k: (torch.tensor(v, dtype=torch.float32) if k in ["labels", "aux_labels"] else torch.tensor(
            v, dtype=torch.int64)) for k, v in batch.items()}

        return batch
