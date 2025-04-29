import pandas as pd
import os
# Load your full CSV
df = pd.read_csv("tedpop/dataset/ted_main_refurbished.csv")

# Shuffle the data first (important)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate split index
split_idx = int(len(df) * 0.9)

# Split
train_df = df[:split_idx]
val_df = df[split_idx:]

# Save to new CSVs
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)

print(f"✅ Train size: {len(train_df)} entries")
print(f"✅ Validation size: {len(val_df)} entries")