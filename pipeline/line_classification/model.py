import transformers
import torch
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from torchmetrics import MeanMetric, MetricCollection


class LineClassificationModel(transformers.LongformerForTokenClassification, pl.LightningModule):

    def __init__(
            self, 
            config: transformers.LongformerConfig, 
            global_attention_id: int = 2,
            global_attend_every_n_line: int = 0
        ):
        super().__init__(config)
        self.global_attention_id = global_attention_id
        self.global_attend_every_n_line = global_attend_every_n_line

        metrics = MetricCollection({
            "accuracy": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "f1": BinaryF1Score()
        })
        self.train_metrics = metrics.clone(prefix="train/")
        self.train_loss = MeanMetric()
        self.val_metrics = metrics.clone(prefix="val/")
        self.val_loss = MeanMetric()

    def forward(
        self, 
        input_ids,
        attention_mask,
        line_sep_token_ids,
    ):
        # Globally attend only to every Nth row
        if self.global_attend_every_n_line > 0:
            global_attention_mask = (input_ids == self.global_attention_id).type(torch.long)
            cumulative_ones = global_attention_mask.cumsum(dim=1)
            positions_to_keep = (cumulative_ones % self.global_attend_every_n_line == 1) & global_attention_mask.bool()
            global_attention_mask = attention_mask * positions_to_keep.int()
        else:
            global_attention_mask = None

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )

        sequence_output = outputs[0]  # [batch, seq, hidden_size]

        # We only consider logits at line_sep_token_ids
        sequence_output = self.dropout(sequence_output)

        # Slice out line sep tokens and run classifier only on those
        batch_size, seq_len = input_ids.shape
        sequence_output = torch.nn.utils.rnn.pad_sequence(
            [sequence_output[i, line_sep_token_ids[i], :] for i in range(batch_size)], 
            batch_first=True
        )
        logits = self.classifier(sequence_output)

        # Return list of tensors and remove padded predictions
        predictions = [
           logits[i, :line_sep_token_ids[i].shape[0], 0]  # [lines]
           for i in range(batch_size)
        ]

        return predictions
    
    def _compute_loss_and_update_metrics(self, batch, predictions, metrics):
        batch_size, _ = batch["input_ids"].shape
        loss = 0.0
        for i in range(batch_size):
            line_labels = torch.zeros_like(batch["line_sep_token_ids"][i])
            if batch["included_lines"][i].sum() > 0:
                line_labels[batch["included_lines"][i]] = 1

            loss += torch.nn.functional.binary_cross_entropy_with_logits(
               input=predictions[i],
               target=line_labels.type(predictions[i].dtype)
            )
            
            if predictions[i].shape[0] > 0:
                metrics(torch.nn.functional.sigmoid(predictions[i]), line_labels)

        loss = loss / batch_size
        return loss
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-5)
        return {
            "optimizer": optim,
        }

    def training_step(self, batch):
        predictions = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            line_sep_token_ids=batch["line_sep_token_ids"],
        )
        
        loss = self._compute_loss_and_update_metrics(batch, predictions, self.train_metrics)
        self.log("train/loss", loss, prog_bar=True)
        self.log_dict(self.train_metrics, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        predictions = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            line_sep_token_ids=batch["line_sep_token_ids"],
        )
        loss = self._compute_loss_and_update_metrics(batch, predictions, self.val_metrics)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log_dict(self.val_metrics, on_epoch=True, on_step=False)
        return loss
    
    def predict_step(self, batch):
        predictions = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            line_sep_token_ids=batch["line_sep_token_ids"],
        )
        return predictions
