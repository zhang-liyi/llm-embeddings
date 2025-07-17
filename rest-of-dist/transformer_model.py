import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import numpy as np
import random

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)

class TransformerModel(torch.nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float=0.5, use_pos=False, continuous_input=False):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.linear = torch.nn.Linear(d_model, ntoken)
        self.use_pos = use_pos
        self.continuous_input = continuous_input
        if not continuous_input:
            self.embedding = torch.nn.Embedding(ntoken, d_model)
        else:
            self.embedding = torch.nn.Linear(ntoken, d_model)
        if self.use_pos:
            self.pos_encoder = PositionalEncoding(d_model, dropout)

    def forward(self, src, memory=None, src_mask=None):
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[batch_size, seq_len, ntoken]``
        """
        src = self.embedding(src) * np.sqrt(self.d_model)
        if self.use_pos:
            src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        self.doc_embd = output
        output = self.linear(output)
        return output

def get_mask(sz, device, mlm=False):
    
    if not mlm:
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask, []
    
    else:
        mask_idx = random.sample(list(range(499)), 75)

        mask = np.zeros((sz, sz))
        mask[:, mask_idx] = float('-inf')

        mask = torch.tensor(mask, device=device, dtype=torch.float32)

        return mask, mask_idx