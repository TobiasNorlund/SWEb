from pipeline.line_classification.model import LineClassificationModel
from pipeline.line_classification.data_loading import IterableLineClassificationDataset, collator
from pipeline.utils import group
from pathlib import Path
from torch.utils.data import DataLoader
import json
import pytorch_lightning as pl
import torch
import transformers
import logging
from defaultlist import defaultlist


# Suppress annoying warning regarding padding
logging.getLogger("transformers.models.longformer.modeling_longformer").setLevel(logging.ERROR)


def move_to_device(val, device):
    if type(val) is torch.Tensor:
        return val.to(device)
    elif type(val) is list:
        return [move_to_device(v, device) for v in val]
    elif type(val) is dict:
        return {k: move_to_device(v, device) for k, v in val.items()}
    else:
        return val
    

class PredictCallback(pl.Callback):

    def __init__(self, overlap: int = 512):
        self.overlap = overlap
        self.line_sep_token_ids = defaultlist(lambda: torch.tensor([], dtype=torch.float32))
        self.predictions = defaultlist(lambda: torch.tensor([], dtype=torch.float32))
        self.losses = defaultlist(lambda: torch.tensor([], dtype=torch.float32))

    def on_predict_batch_end(self, trainer, model, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        predictions = outputs  # List[Tensor[float]]
        batch_size, _ = batch["input_ids"].shape

        predictions = move_to_device(predictions, "cpu")
        batch = move_to_device(batch, "cpu")

        for i in range(batch_size):
            if self.predictions[batch["doc_idx"][i]].shape[0] == 0:
                # Use all predictions for first window
                self.predictions[batch["doc_idx"][i]] = predictions[i].type(torch.float32)
                self.line_sep_token_ids[batch["doc_idx"][i]] = batch["line_sep_token_ids"][i] + batch["token_start_idx"][i]
            else:
                if batch["line_sep_token_ids"][i].shape[0] == 0:
                    # if there are no line sep token in this sequence, skip...
                    continue

                # Find the index (in this token input sequence) of the first line sep token we should use from this sequence 
                split_token_idx = batch["line_sep_token_ids"][i][batch["line_sep_token_ids"][i] >= self.overlap//2]
                if split_token_idx.shape[0] > 0:
                    split_token_idx = split_token_idx[0]
                else:
                    # All line sep tokens are close to the edge. In such case, there are no relevant predictions
                    continue
                    
                # The same index but in the global sequence
                split_token_idx_prev = split_token_idx + batch["token_start_idx"][i]

                # Where in the list of sep tokens (in this token input sequence) is this token?
                split_token_idx_local = (batch["line_sep_token_ids"][i] == split_token_idx).nonzero()[0, 0]
                # Where in the list of sep tokens (in the global sep token list) is this token?
                if split_token_idx_prev in self.line_sep_token_ids[batch["doc_idx"][i]]:
                    split_token_idx_prev_local = (self.line_sep_token_ids[batch["doc_idx"][i]] == split_token_idx_prev).nonzero()[0, 0]
                else:
                    # In case of very long lines, it might not be found. In such case we simply concatenate the predictions
                    split_token_idx_prev_local = self.line_sep_token_ids[batch["doc_idx"][i]].shape[0]

                self.line_sep_token_ids[batch["doc_idx"][i]] = torch.cat([
                    self.line_sep_token_ids[batch["doc_idx"][i]][:split_token_idx_prev_local],
                    batch["line_sep_token_ids"][i][split_token_idx_local:] + batch["token_start_idx"][i]
                ])

                self.predictions[batch["doc_idx"][i]] = torch.cat([
                    self.predictions[batch["doc_idx"][i]][:split_token_idx_prev_local],
                    predictions[i][split_token_idx_local:].type(torch.float32)
                ])

            if batch["included_lines"][i] is not None:
                # If we have labels, compute loss
                line_labels = torch.zeros_like(batch["line_sep_token_ids"][i])
                if batch["included_lines"][i].sum() > 0:
                    line_labels[batch["included_lines"][i]] = 1

                losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    input=predictions[i],
                    target=line_labels.type(predictions[i].dtype),
                    reduction="none"
                ).type(torch.float32)

                if self.losses[batch["doc_idx"][i]].shape[0] == 0:
                    self.losses[batch["doc_idx"][i]] = losses.type(torch.float32)
                else:
                    self.losses[batch["doc_idx"][i]] = torch.cat([
                        self.losses[batch["doc_idx"][i]][:split_token_idx_prev_local],  # Discard `overlap` / 2 tail predictions
                        losses[split_token_idx_local:].type(torch.float32)  # Discard `overlap` / 2 head predictions
                    ])


def main(args):
    import gzip
    import logging

    logging.basicConfig(level=logging.INFO)

    logging.info(f"Loading data from {args.input}")
    f = gzip.open(args.input) if args.input.endswith("gz") else open(args.input)
    with f:
        data = [json.loads(line) for line in f]

    # Sort on token length
    tokenizer = transformers.AutoTokenizer.from_pretrained("severinsimmler/xlm-roberta-longformer-base-16384")
    doc_token_lens = [len(tokens) for b in group(data, group_size=256) for tokens in tokenizer.batch_encode_plus([ex["content"] for ex in b]).input_ids]
    sorted_with_index = sorted(enumerate(zip(data, doc_token_lens)), key=lambda x: x[1][1])
    data = [ex for _, (ex, _) in sorted_with_index]
    original_indices = [index for index, _ in sorted_with_index]  # index to orig list
    new_index = {prev_ids: new_idx for new_idx, prev_ids in enumerate(original_indices)}

    test_ds = IterableLineClassificationDataset(data, tokenizer)
    test_dl = DataLoader(test_ds, batch_size=16, collate_fn=collator, num_workers=1)

    config = transformers.AutoConfig.from_pretrained("severinsimmler/xlm-roberta-longformer-base-16384")
    config.num_labels = 1
    model = LineClassificationModel.load_from_checkpoint(
        Path(args.checkpoint),
        config=config,
        map_location=torch.device("cpu")
    )

    callback = PredictCallback()

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[5],
        callbacks=[callback],
        precision="bf16-mixed",
    )

    trainer.predict(model, dataloaders=test_dl)

    with open(args.output, "w") as f:
        for i in range(len(data)):
            pred = callback.predictions[new_index[i]]
            ex = data[new_index[i]]
            # Make sure the logic for merging predictions for long documents seems to work
            # We want one prediction for each newline
            assert pred.shape[0] == ex["content"].count("\n")
            ex["line_predictions"] = torch.nn.functional.sigmoid(pred).tolist()
            json.dump(ex, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    main(args)