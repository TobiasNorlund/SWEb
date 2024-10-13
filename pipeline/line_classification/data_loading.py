import torch
import math
from typing import List


def batch(iterable, n):
    group = []
    for x in iterable:
        group.append(x)
        if len(group) == n:
            yield group
            group = []
    if group:
        yield group


class IterableLineClassificationDataset(torch.utils.data.IterableDataset):
    """
    Iterable version of LineClassificationDataset, that support long documents (longer than model context len)
    by splitting them up.

    Note: Only to be used for inference, as it currently does not take any line labels required for training 
    """

    def __init__(
        self,
        documents: List[str],
        tokenizer,
        overlap=512,
    ):
        super().__init__()
        self.documents = documents
        self.tokenizer = tokenizer
        self.overlap = overlap

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            start = 0
            end = len(self.documents)
        else:
            per_worker = int(math.ceil(len(self.documents) / float(worker_info.num_workers)))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.documents))
        
        max_len = self.tokenizer.model_max_length
        for doc_idx in range(start, end):
            text = self.documents[doc_idx]
            text += "\n" if not text.endswith("\n") else "" # Add trailing newline if not already there
            text = text.replace(self.tokenizer.sep_token, "")  # Remove any existing sep token occurrences before replacing newlines
            text = text.replace("\n", self.tokenizer.sep_token)  # Replace newlines with special sep token
            tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False, truncation=False)
            for token_start_idx in range(0, tokens.input_ids.shape[1], max_len-self.overlap if tokens.input_ids.shape[1] > max_len else max_len):
                input_ids = tokens.input_ids[0, token_start_idx : token_start_idx + max_len]
                attention_mask = tokens.attention_mask[0, token_start_idx : token_start_idx + max_len]
                line_sep_token_ids = (input_ids == self.tokenizer.sep_token_id).nonzero()[:, 0]
                ex = {
                    "doc_idx": torch.tensor(doc_idx),
                    "token_start_idx": torch.tensor(token_start_idx),
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "line_sep_token_ids": line_sep_token_ids
                }

                if line_sep_token_ids.shape[0] > 0:
                    # No need to yield if there are no line separators, since then there is nothing to predict 
                    yield ex


class LineClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, documents, tokenizer):
        super().__init__()
        self.documents = documents
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        tokens = self.tokenizer(self.documents[index]["content"].replace("\n", self.tokenizer.sep_token), return_tensors="pt", add_special_tokens=False, truncation=True)
        line_sep_token_ids = (tokens.input_ids[0] == self.tokenizer.sep_token_id).nonzero()[:, 0]
        
        ex = {
            "doc_idx": torch.tensor(index),
            "token_start_idx": torch.tensor(0),  # TODO: Support splitting up long documents? 
            "input_ids": tokens.input_ids[0],
            "attention_mask": tokens.attention_mask[0],
            "line_sep_token_ids": line_sep_token_ids,
        }
        
        if "selected_lines" in self.documents[index]:
            ex["included_lines"] = torch.tensor([
                line_no 
                for line_no in self.documents[index]["selected_lines"]
                if line_no < line_sep_token_ids.shape[0]  # Don't include truncated lines
            ])
        return ex


def collator(batch):
    max_len = max(ex["input_ids"].shape[0] for ex in batch)
    input_ids = torch.stack([
        torch.nn.functional.pad(ex["input_ids"], (0, max_len - ex["input_ids"].shape[0]))
        for ex in batch
    ])
    attention_mask = torch.stack([
        torch.nn.functional.pad(ex["attention_mask"], (0, max_len - ex["attention_mask"].shape[0]))
        for ex in batch
    ])
    line_sep_token_ids = [ex["line_sep_token_ids"] for ex in batch]
    included_lines = [ex["included_lines"] if "included_lines" in ex else None for ex in batch]

    return {
        "doc_idx": torch.stack([ex["doc_idx"] for ex in batch]),
        "token_start_idx": torch.stack([ex["token_start_idx"] for ex in batch]),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "line_sep_token_ids": line_sep_token_ids,
        "included_lines": included_lines
    }
