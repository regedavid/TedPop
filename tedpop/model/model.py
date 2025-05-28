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
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  
        self.output_dim = output_dim

    def forward(self, audio, return_sequence=False):
        """
        audio: [B, 13, T]
        return_sequence: if True, return [B, T', D] for cross-attn
        """
        x = self.cnn(audio)  # [B, D, T']
        if return_sequence:
            return x.transpose(1, 2)  # → [B, T', D] for attention
        else:
            x = self.pool(x)         # [B, D, 1]
            return x.squeeze(-1)     # [B, D]  # expects shape (batch, 13, T)
   
 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)  # [B, T, D]
        return x + self.pe[:, :seq_len, :]


class TransformerTextEncoder(nn.Module):
    def __init__(self, vocab_size, dim_model=512, nhead=8, num_layers=12, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.output_dim = dim_model
        self.embedding = nn.Embedding(vocab_size, dim_model)
        self.pos_enc = PositionalEncoding(dim_model)
        encoder_layer = nn.TransformerEncoderLayer(dim_model, nhead, dim_ff, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, src_ids, src_mask=None):
        x = self.embedding(src_ids) # [B, T, D]
        x = self.pos_enc(x) # [B, T, D]
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_mask) # [B, T, D]
        pooled = encoded.mean(dim=1) # [B, D] - global average pooling
        
        return pooled
    
    def get_sequence(self, src_ids, src_mask=None):
        x = self.embedding(src_ids) # [B, T, D]
        x = self.pos_enc(x)
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_mask)
        return encoded  # [B, T, D]
    

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context, context_mask=None):
        # query, context: [B, T, D]
        # context_mask: [B, T] -> padding mask for context
        attn_output, _ = self.attn(query, context, context, key_padding_mask=context_mask)
        return self.norm(query + self.dropout(attn_output))
    

class BidirectionalCrossAttention(nn.Module):
    def __init__(self, text_dim, audio_dim, shared_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, shared_dim)
        self.audio_proj = nn.Linear(audio_dim, shared_dim)

        self.text_to_audio = CrossAttentionBlock(shared_dim, num_heads, dropout)
        self.audio_to_text = CrossAttentionBlock(shared_dim, num_heads, dropout)
        self.shared_dim = shared_dim

    def forward(self, text_seq, audio_seq, audio_mask=None, text_mask=None):
        # Project to shared dim
        text = self.text_proj(text_seq)   # [B, T_text, D]
        audio = self.audio_proj(audio_seq) # [B, T_audio, D]

        # Cross-attention
        text_attn = self.text_to_audio(text, audio, context_mask=audio_mask)
        audio_attn = self.audio_to_text(audio, text, context_mask=text_mask)

        return text_attn, audio_attn