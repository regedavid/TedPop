from transformers import AutoTokenizer
import torch
# Load encoders
from tedpop.model.model import TextEncoder
from tedpop.model.model import AudioEncoder
from tedpop.train.train import TEDRegressor
from tedpop.dataset.dataset import TEDDataset, TEDMultimodalDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load from checkpoint
checkpoint_path = "lightning_logs/version_0/checkpoints/epoch=27-step=1876.ckpt"

# Rebuild model structure
text_encoder = TextEncoder("bert-base-uncased")
audio_encoder = None  # Or None for text-only
model = TEDRegressor.load_from_checkpoint(
    checkpoint_path,
    text_encoder=text_encoder,
    audio_encoder=audio_encoder
)
model = model.to(device)
model.eval()

# Text-only
val_dataset = TEDDataset(
    csv_file="tedpop/dataset/val_filtered.csv",
    text_column="transcript",
    target_column="viewCount",
    transform_target='log'
)

example = val_dataset[2]  # or any other index
print("Target (log view count):", example["target"])
print("Target view count:", int(torch.expm1(example["target"]).item()))

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

inputs = tokenizer(
    example["text"],
    return_tensors="pt",
    truncation=True,
    max_length=512
)

input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

if "audio" in example:
    mfcc_tensor = example["audio"].unsqueeze(0)  # shape: (1, 13, T)
else:
    mfcc_tensor = None

with torch.no_grad():
    pred = model(input_ids=input_ids, attention_mask=attention_mask, audio=mfcc_tensor)

# Inverse transform
pred_viewcount = torch.expm1(pred).item()

print(f"Predicted view count: {int(pred_viewcount):,}")
print(f"Predicted log view count: {int(pred):,}")