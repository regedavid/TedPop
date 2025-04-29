import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from pytorch_lightning import Trainer

from tedpop.dataset.dataset import TEDDataset, collate_fn 

class TEDRegressor(pl.LightningModule):
    def __init__(self, model_name="bert-base-uncased", lr=2e-5):
        super().__init__()
        self.save_hyperparameters()

        self.bert = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token output
        preds = self.regressor(cls_output)
        return preds.squeeze(-1)  # [batch_size]

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]

        preds = self(input_ids, attention_mask)
        loss = self.loss_fn(preds, targets)

        # Calculate RMSE on normal scale
        preds_exp = torch.expm1(preds)
        targets_exp = torch.expm1(targets)
        rmse = torch.sqrt(torch.mean((preds_exp - targets_exp) ** 2))

        self.log("train_loss", loss)
        self.log("train_rmse", rmse)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]

        preds = self(input_ids, attention_mask)
        loss = self.loss_fn(preds, targets)

        preds_exp = torch.expm1(preds)
        targets_exp = torch.expm1(targets)
        rmse = torch.sqrt(torch.mean((preds_exp - targets_exp) ** 2))

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_rmse", rmse, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

if __name__ == "__main__":
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load datasets
    train_dataset = TEDDataset(
        csv_file="tedpop/dataset/train.csv",
        text_column="transcript",
        target_column="viewCount",
        transform_target=lambda x: torch.log1p(torch.tensor(x))
    )

    val_dataset = TEDDataset(
        csv_file="tedpop/dataset/val.csv",
        text_column="transcript",
        target_column="viewCount",
        transform_target=lambda x: torch.log1p(torch.tensor(x))
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda batch: collate_fn(batch, tokenizer))
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=lambda batch: collate_fn(batch, tokenizer))

    model = TEDRegressor(model_name="bert-base-uncased", lr=2e-5)

    trainer = Trainer(
        max_epochs=5,
        accelerator="gpu",  
        devices=1,          
        precision=16,       
    )

    trainer.fit(model, train_loader, val_loader)