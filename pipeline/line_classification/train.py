from pathlib import Path
import pytorch_lightning as pl
import transformers
import json
import logging
from data_loading import LineClassificationDataset, collator
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from model import LineClassificationModel


# Suppress annoying warning regarding padding
logging.getLogger("transformers.models.longformer.modeling_longformer").setLevel(logging.ERROR)
    

def load_annotated_data(file):
    data = []
    with open(file) as f:
        for line in f:
            parsed = json.loads(line)
            if  "is_ignored" not in parsed or parsed["is_ignored"] != True:
                data.append(parsed)
    return data


def main(args):
    data = []
    for file in args.data_batches:
        data += load_annotated_data(file)

    tokenizer = transformers.AutoTokenizer.from_pretrained("severinsimmler/xlm-roberta-longformer-base-16384")

    val_documents = data[:100]
    train_documents = data[100:]

    train_ds = LineClassificationDataset(train_documents, tokenizer)
    val_ds = LineClassificationDataset(val_documents, tokenizer)

    train_dl = DataLoader(train_ds, batch_size=1, collate_fn=collator)
    val_dl = DataLoader(val_ds, batch_size=1, collate_fn=collator)

    model = LineClassificationModel.from_pretrained(
        "severinsimmler/xlm-roberta-longformer-base-16384", 
        global_attend_every_n_line=args.global_attend_every_n_line,
        id2label={0: "include"}, 
        label2id={"include": 0},
        num_hidden_layers=args.num_layers,
        attention_window=[256] * args.num_layers
    )

    early_stop_callback = EarlyStopping(
        monitor='val/f1',
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode='max'
    )

    wandb_logger = WandbLogger(
        group="markdown-extraction/train",
        project='common-crawl',
    )
    wandb_logger.experiment.config.update({
        "data_batches": args.data_batches,
        "num_layers": args.num_layers,
        "global_attend_every_n_line": args.global_attend_every_n_line,
    })

    model_checkpoint_callback = ModelCheckpoint(
        dirpath=Path(wandb_logger.experiment.dir) / ".." / "tmp",
        monitor='val/f1',
        save_top_k=1,
        mode='max',
        save_last=False,
        verbose=True,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[args.gpu],
        callbacks=[early_stop_callback, model_checkpoint_callback, LearningRateMonitor(logging_interval='step')],
        logger=wandb_logger,
        log_every_n_steps=50,
        precision="bf16-mixed",
    )

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-batches", nargs="+", default=[
        "annotation_tool/backend/data/data.jsonl"
    ])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--global-attend-every-n-line", type=int, default=0)
    args = parser.parse_args()
    main(args)