import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel

BASE_MODEL = "distilbert-base-uncased"
LORA_DIR = "models/bert_lora_classifier" 

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_label_maps(lora_dir: str):
    cfg_path = os.path.join(lora_dir, "config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
        id2label = cfg.get("id2label")
        label2id = cfg.get("label2id")

        if isinstance(id2label, dict) and len(id2label) > 0:
            fixed_id2label = {int(k): v for k, v in id2label.items()}
            fixed_label2id = None
            if isinstance(label2id, dict) and len(label2id) > 0:
                fixed_label2id = {k: int(v) for k, v in label2id.items()}
            return fixed_id2label, fixed_label2id

    return None, None

id2label, label2id = load_label_maps(LORA_DIR)
if id2label is None:
    raise RuntimeError(
        f"Could not find id2label in {LORA_DIR}/config.json.\n"
        "Fix: copy config.json from your trained model folder OR provide labels list manually."
    )

num_labels = len(id2label)

tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)

# correct fallback mapping: label -> id
if label2id is None:
    label2id = {label: idx for idx, label in id2label.items()}

base_config = AutoConfig.from_pretrained(
    BASE_MODEL,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, config=base_config)
model = PeftModel.from_pretrained(base, LORA_DIR).to(device)
model.eval()

def predict(text: str):
    x = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**x).logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())
        conf = float(torch.max(probs).item())
    return {"label": id2label[pred_id], "confidence": round(conf, 4)}

def classify(text: str):
    out = predict(text)
    return out["label"], out["confidence"]

if __name__ == "__main__":
    print(predict("App crashes when I click submit"))
