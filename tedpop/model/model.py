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
   
 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(0)
        return x + self.pe[:seq_len]


class TransformerTextEncoder(nn.Module):
    def __init__(self, vocab_size, dim_model=512, nhead=8, num_layers=12, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.output_dim = dim_model
        self.embedding = nn.Embedding(vocab_size, dim_model)
        self.pos_enc = PositionalEncoding(dim_model)
        encoder_layer = nn.TransformerEncoderLayer(dim_model, nhead, dim_ff, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, src_ids, src_mask=None):
        x = self.embedding(src_ids).transpose(0, 1)
        x = self.pos_enc(x)
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_mask)
        pooled = encoded.mean(dim=0)
        
        return pooled