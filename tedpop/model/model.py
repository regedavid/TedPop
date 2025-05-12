from transformers import AutoModel
import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.output_dim = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        return cls_output
    
class AudioEncoder(nn.Module):
    def __init__(self, input_dim=13, output_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # → (batch, 64, 1)
            nn.Flatten()              # → (batch, 64)
        )
        self.output_dim = output_dim

    def forward(self, audio):
        return self.cnn(audio)  # expects shape (batch, 13, T)