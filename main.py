import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import (
    BertModel,
    BertTokenizerFast as BertTokenizer,
    get_linear_schedule_with_warmup,
)

BERT_MODEL_NAME = 'bert-base-cased'
MAX_TOKEN_LEN = 512
N_EPOCHS = 10
BATCH_SIZE = 4
LABEL_COLUMNS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

class CourtCasesDataset(Dataset):

    def __init__(self, 
            dataset: list[dict], 
            tokenizer,
            max_token_len
            ):

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        text = " ".join(self.dataset[idx]["text"])
        
        encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_token_len,
                return_token_type_ids=False,
                padding="max_length",
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
                )

        Y = self.dataset[idx]["labels"]

        y = torch.zeros(10)
        for article in Y:
            y[article] = 1

        return dict(
                text = text,
                input_ids = encoding["input_ids"].flatten(),
                attention_mask = encoding["attention_mask"].flatten(),
                labels = y
                )

class CourtCasesDataModule(pl.LightningDataModule):

    def __init__(self, 
            train_set, 
            test_set, 
            valid_set, 
            tokenizer,
            batch_size=8,
            max_token_len=512
            ):
        super().__init__()
        self.train_set     = train_set
        self.test_set      = test_set
        self.valid_set     = valid_set
        self.tokenizer     = tokenizer
        self.batch_size    = batch_size
        self.max_token_len = max_token_len
        
    def setup(self, stage = None):
        self.train_dataset = CourtCasesDataset(
                self.train_set,
                self.tokenizer,
                self.max_token_len
                )

        self.test_dataset = CourtCasesDataset(
                self.test_set,
                self.tokenizer,
                self.max_token_len
                )

        self.valid_dataset = CourtCasesDataset(
                self.valid_set,
                self.tokenizer,
                self.max_token_len
                )

    def train_dataloader(self):
        return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4
                )

    def test_dataloader(self):
        return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=4
                )

    def val_dataloader(self):
        return DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size,
                num_workers=4
                )

class CourtCaseClassifier(pl.LightningModule):

    def __init__(self, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, 
                return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 10)
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    # def training_epoch_end(self, outputs):

        # labels = []
        # predictions = []

        # for output in outputs:
            # for out_labels in output["labels"].detach().cpu():
                # labels.append(out_labels)
            # for out_predictions in output["predictions"].detach().cpu():
                # predictions.append(out_predictions)

        # labels = torch.stack(labels).int()
        # predictions = torch.stack(predictions)

        # for i, name in enumerate(LABEL_COLUMNS):
            # print(i)
            # class_roc_auc = auroc(predictions[:, i], labels[:, i])
            # print(class_roc_auc)
            # self.logger.experiment.add_scalar(
                    # f"{name}_roc_auc/Train", class_roc_auc, 
                    # self.current_epoch)

    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=2e-5)

        scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.n_warmup_steps,
                num_training_steps = self.n_training_steps
                )
        return dict(
                optimizer=optimizer,
                lr_scheduler=dict(
                    scheduler=scheduler,
                    interval='step'
                    )
                )

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--action', 
            type=str, 
            choices=['train', 'test'],
            help="choose action to be performed ('train' or 'test')",
            required=True
            )
    parser.add_argument(
            '--path', 
            type=str, 
            help='relative path to trained model'
            )
    args = parser.parse_args()
    ACTION = args.action
    MODEL_PATH = args.path

    from datasets import load_dataset

    dataset = load_dataset("lex_glue", "ecthr_a")

    # # all train, test and valid cases with labels 
    # train_set = dataset["train"]
    # test_set = dataset["test"]
    # valid_set = dataset["validation"]

    # Uncomment for a reduced dataset size
    size = 210
    train_set = [train_set[i] for i in range(size)]
    train_set = [test_set[i] for i in range(size)]
    train_set = [valid_set[i] for i in range(size)]

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    data_module = CourtCasesDataModule(
            train_set,
            test_set,
            valid_set,
            tokenizer,
            batch_size=BATCH_SIZE,
            max_token_len=MAX_TOKEN_LEN
            )

    steps_per_epoch = len(train_set) // BATCH_SIZE
    total_training_steps = steps_per_epoch * N_EPOCHS

    warmup_steps = total_training_steps // 5

    model = CourtCaseClassifier(
            n_warmup_steps=warmup_steps,
            n_training_steps=total_training_steps
            )

    checkpoint_callback = ModelCheckpoint(
            dirpath='checpoints',
            filename="best_checkpoint",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode='min'
            )

    logger = TensorBoardLogger("lightning_logs", name="violated-articles")

    early_stopping_callback = EarlyStopping(monitor='val_loss',
            patience=2)

    trainer = pl.Trainer(
            logger=True,
            enable_checkpointing=True,
            callbacks=[early_stopping_callback],
            max_epochs=N_EPOCHS,
            accelerator='gpu',
            devices=1,
            enable_progress_bar=True
            )

    # FIT A MODEL TO TRAINING DATA
    if ACTION == 'train':
        trainer.fit(model, data_module)

    # # LOAD TRAINED MODEL
    if ACTION == 'test':
        if MODEL_PATH == None:
            raise ValueError(
        "Path to trained model has not been given run with --h for help"
        )
        trained_model = CourtCaseClassifier.load_from_checkpoint(MODEL_PATH)

        trained_model.eval()
        trained_model.freeze()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trained_model = trained_model.to(device)

        val_dataset = CourtCasesDataset(
                valid_set,
                tokenizer,
                max_token_len=MAX_TOKEN_LEN
                )

        test_dataset = CourtCasesDataset(
                test_set,
                tokenizer,
                max_token_len=MAX_TOKEN_LEN
                )

        def prepare_preds_and_labels(dataset):
            """Passes a dataset (validation or test sets) through the trained 
            model and returns the predictions and labels"""

            predictions = []
            labels = []

            print("Passing dataset through trained model")
            for i, item in enumerate(dataset):
                if i%100 == 0:
                    print("Examples passed:", i)
                _, prediction = trained_model(
                        item["input_ids"].unsqueeze(dim=0).to(device),
                        item["attention_mask"].unsqueeze(dim=0).to(device)
                        )
                predictions.append(prediction.flatten())
                labels.append(item["labels"].int())
                
            predictions = torch.stack(predictions).detach().cpu()
            labels = torch.stack(labels).detach().cpu()

            return predictions, labels

        val_preds, val_labels = prepare_preds_and_labels(val_dataset)
        test_preds, test_labels = prepare_preds_and_labels(test_dataset)

        import pickle
        with open('val_preds_labels.pickle', 'wb') as v_pkl:
            pickle.dump([val_preds, val_labels], v_pkl)

        with open('test_preds_labels.pickle', 'wb') as t_pkl:
            pickle.dump([test_preds, test_labels], t_pkl)
