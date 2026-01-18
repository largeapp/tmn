from torch import nn, Tensor
import torch
import math
from einops import rearrange


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        pe = scale_factor * pe.unsqueeze(0)     # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        # x = x + self.pe[:x.size(0), :]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        # self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model))  # requires_grad automatically set to True
        # nn.init.uniform_(self.pe, -0.02, 0.02)
        nn.init.normal_(self.pe, std=.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        # x = x + self.pe[:x.size(0), :]

        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding
    elif pos_encoding == 'none':
        return nn.Identity

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed'/'none', not '{}'".format(pos_encoding))


"""
reference1: https://github.com/aju22/RoPE-PyTorch/blob/main/RoPE.ipynb
reference2: https://github.com/T-Li-1/rotary-positional-embedding/blob/main/RoPE.py
reference3: hugging face transformer.models.llama.modeling_llama.LlamaRotaryEmbedding
"""
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=1024, base=10_000) -> None:
        super().__init__()
        self.max_seq_len_cached = None

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(seq_len=max_seq_len,
                                device=self.inv_freq.device)

    def _set_cos_sin_cache(self, seq_len, *, device):
        self.max_seq_len_cached = seq_len
        
        seq = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freq = torch.einsum("i,j->ij", seq, self.inv_freq)
        pos_embed = torch.cat((freq, freq), dim=-1)
        self.register_buffer('cos_cached', pos_embed.cos(), persistent=False)   # [max_seq_len, head_dim]
        self.register_buffer('sin_cached', pos_embed.sin(), persistent=False)
    
    def rotate_half(self, x):
        # x's the number of last channel is even number
        x = rearrange(x, "... (j d) -> ... j d", j=2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, *qk:tuple):
        # apply rotary positional embedding to Q and K
        # assert len(qk[0].shape) == 4, 'before matual, the shape should be [B, H, L, D]'
        *_, L, D = qk[0].shape

        if L > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=L, device=qk[0].device)

        cos = self.cos_cached[:L]
        sin = self.sin_cached[:L]
        return tuple(map(lambda x: x*cos+self.rotate_half(x)*sin, qk))
    
"""
example:
rope = RotaryEmbedding(D).to(x.device)
q, k = rope(q, k)
"""