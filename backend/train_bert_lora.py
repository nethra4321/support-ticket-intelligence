import os
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

from peft import LoraConfig, get_peft_model, TaskType

PARQUET_PATH = "../data/twitter_train_parquet"
BASE_MODEL = "distilbert-base-uncased"
OUT_DIR = "models/bert_lora_classifier"

MAX_LEN = 192          
EPOCHS = 1            
LR = 2e-4
BATCH = 32             
SEED = 42

MAX_SAMPLES = 20_000   
MIN_LABEL_COUNT = 200  


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()


    vc = df["label"].value_counts()
    keep = vc[vc >= MIN_LABEL_COUNT].index
    df = df[df["label"].isin(keep)].copy()

    if len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=SEED).reset_index(drop=True)

    labels = sorted(df["label"].unique().tolist())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    df["label_id"] = df["label"].map(label2id).astype(int)

    ds = Dataset.from_pandas(df[["text", "label_id"]], preserve_index=False)
    split = ds.train_test_split(test_size=0.2, seed=SEED)
    train_ds, eval_ds = split["train"], split["test"]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

    train_ds = train_ds.map(tok, batched=True, remove_columns=["text"])
    eval_ds = eval_ds.map(tok, batched=True, remove_columns=["text"])

    train_ds = train_ds.rename_column("label_id", "labels")
    eval_ds = eval_ds.rename_column("label_id", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=OUT_DIR,
        learning_rate=LR,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        num_train_epochs=EPOCHS,

        eval_strategy="epoch",
        save_strategy="no",

        logging_steps=50,
        report_to="none",
        fp16=False,  
        seed=SEED,

        dataloader_pin_memory=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    print("Saved model to:", OUT_DIR)
    print("Labels:", labels)
    print(f"Trained on {len(df):,} samples, MAX_LEN={MAX_LEN}, BATCH={BATCH}, EPOCHS={EPOCHS}")


if __name__ == "__main__":
    main()
