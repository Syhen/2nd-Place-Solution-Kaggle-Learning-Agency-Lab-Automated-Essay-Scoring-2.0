"""
@created by: heyao
@created at: 2024-06-05 00:18:48
From Raja
"""
import pandas as pd
from datasets import Dataset
from nltk import sent_tokenize

NEW_TOKENS = [
    "[=sop=]",
    "[=eop=]",
    "[=sos=]",
    "[=eos=]",
]


class AES2Dataset:
    """Dataset class for AES2 task"""

    def __init__(self, cfg, tokenizer):
        # assign config
        self.cfg = cfg

        # label columns
        self.target_name = "score"

        # load tokenizer
        self.tokenizer = tokenizer
        self.load_tokenizer()

    def load_tokenizer(self):
        # additional tokens
        # start/end of paragraphs
        self.sop_id = self.tokenizer.convert_tokens_to_ids(NEW_TOKENS[0])
        self.eop_id = self.tokenizer.convert_tokens_to_ids(NEW_TOKENS[1])
        # start/end of sentences
        self.sos_id = self.tokenizer.convert_tokens_to_ids(NEW_TOKENS[2])
        self.eos_id = self.tokenizer.convert_tokens_to_ids(NEW_TOKENS[3])

    def get_segments(self, text):
        paragraphs = text.split("\n\n")
        to_return = []
        for para in paragraphs:
            sents = sent_tokenize(para)
            to_return.append(sents)
        return to_return

    def pre_process(self, df):
        def decorate(text):
            segments = self.get_segments(text)
            to_return = ""
            for para in segments:
                wrapped_para = ""
                for sent in para:
                    if len(sent.strip()) <= 2:
                        continue
                    wrapped_para += f"[=sos=]{sent}[=eos=]"
                if len(wrapped_para) >= 16:
                    to_return += f"[=sop=]{wrapped_para}[=eop=]\n\n"
            return to_return

        df["full_text"] = df["full_text"].apply(decorate)

        return df

    def tokenize_function(self, examples):
        tz = self.tokenizer(
            examples["full_text"],
            padding=False,
            truncation=True,
            max_length=self.cfg.train.max_length,
            add_special_tokens=True,
            return_token_type_ids=True,
        )
        return tz

    def process_spans(self, examples):
        span_head_idxs, span_tail_idxs = [], []
        sentence_head_idxs, sentence_tail_idxs = [], []

        for ex_ids in examples["input_ids"]:
            ex_len = len(ex_ids)
            head_buffer = min(8, ex_len - 2)
            ex_ids = ex_ids[:ex_len - head_buffer]

            ex_head_idxs = [pos for pos, this_id in enumerate(ex_ids) if this_id == self.sop_id]
            ex_tail_idxs = [pos for pos, this_id in enumerate(ex_ids) if this_id == self.eop_id]

            ex_s_head_idxs = [pos for pos, this_id in enumerate(ex_ids) if this_id == self.sos_id]
            ex_s_tail_idxs = [pos for pos, this_id in enumerate(ex_ids) if this_id == self.eos_id]

            if len(ex_tail_idxs) != len(ex_head_idxs):
                ex_tail_idxs.append(ex_len - 1)  # SEP token

            if len(ex_s_tail_idxs) != len(ex_s_head_idxs):
                ex_s_tail_idxs.append(ex_len - 1)  # SEP token

            span_head_idxs.append(ex_head_idxs)
            span_tail_idxs.append(ex_tail_idxs)

            sentence_head_idxs.append(ex_s_head_idxs)
            sentence_tail_idxs.append(ex_s_tail_idxs)

        return {
            "span_head_idxs": span_head_idxs,
            "span_tail_idxs": span_tail_idxs,
            "sentence_head_idxs": sentence_head_idxs,
            "sentence_tail_idxs": sentence_tail_idxs,
        }

    def sanity_check_head_tail(self, examples):
        for head_idxs, tail_idxs in zip(examples["span_head_idxs"], examples["span_tail_idxs"]):
            assert len(head_idxs) == len(tail_idxs), f"heads: {head_idxs}, tails: {tail_idxs}"
            for head, tail in zip(head_idxs, tail_idxs):
                assert tail > head + 1, f"span heads: {head_idxs}, span tails: {tail_idxs}"

        for head_idxs, tail_idxs in zip(examples["sentence_head_idxs"], examples["sentence_tail_idxs"]):
            assert len(head_idxs) == len(tail_idxs), f"heads: {head_idxs}, tails: {tail_idxs}"
            for head, tail in zip(head_idxs, tail_idxs):
                assert tail > head + 1, f"sent heads: {head_idxs}, sent tails: {tail_idxs}"

    def generate_labels(self, examples):
        labels = []
        for val in examples["score"]:
            labels.append(val)
        return {"labels": labels}

    def compute_input_length(self, examples):
        return {"input_length": [len(x) for x in examples["input_ids"]]}

    def get_dataset(self, df, mode='train'):
        """main api for creating the Feedback dataset
        :param df: input annotation dataframe
        :type df: pd.DataFrame
        :param mode: check if required for train or infer, defaults to 'train'
        :type mode: str, optional
        :return: the created dataset
        :rtype: Dataset
        """
        assert mode in ["train", "infer"], "mode must in train or infer."
        df = self.pre_process(df)

        print(f"sample text:")
        print("==" * 40)
        print(df.sample().full_text.values[0])
        print("==" * 40)

        task_dataset = Dataset.from_pandas(df)
        task_dataset = task_dataset.map(self.tokenize_function, batched=True)
        task_dataset = task_dataset.map(self.process_spans, batched=True)
        task_dataset = task_dataset.map(self.sanity_check_head_tail, batched=True)
        task_dataset = task_dataset.map(self.compute_input_length, batched=True)

        if mode != "infer":
            task_dataset = task_dataset.map(self.generate_labels, batched=True)

        try:
            task_dataset = task_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            print(e)

        # ---
        print("##" * 20)
        print(task_dataset[0])
        print("##" * 20)

        return task_dataset


if __name__ == '__main__':
    import pandas as pd
    from omegaconf import OmegaConf, DictConfig
    from transformers import AutoTokenizer, AddedToken

    config = DictConfig(content={})
    config.train = {}
    config.train.max_length = 1024
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    tokenizer.add_tokens(AddedToken("\n\n", normalized=False))
    for token in NEW_TOKENS:
        tokenizer.add_tokens(AddedToken(token, normalized=False))
    df = pd.read_csv(
        "/Users/heyao/projects/kaggle-aes2/my_datasets/learning-agency-lab-automated-essay-scoring-2/train.csv")
    df["full_text"] = df["full_text"].apply(lambda x: x.strip())
    ds = AES2Dataset(config, tokenizer)
    ds = ds.get_dataset(df, mode="train")
    print(ds[0])
