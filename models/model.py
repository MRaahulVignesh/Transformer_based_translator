import torch 
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding_layer = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding_layer(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        ## Build the positional encoding tensor
        pe = torch.zeros(seq_len, d_model) # (Seq_Len, d_model)
        pos = torch.arange(start=0, end=seq_len).unsqueeze(1) # (Seq_Len, 1)
        div = torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float)/d_model).float() # (d_model//2,)

        # (Seq_Len, 1) / (d_model//2,) => (Seq_Len, d_model//2) / (1, d_model//2)
        pe[:, ::2] = torch.sin(pos/div)
        pe[:, 1::2] = torch.cos(pos/div)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.size(1), :].detach())
        return self.dropout(x)
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        assert d_model%num_heads == 0, "d_model is not divisible num_heads"

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = self.d_model//num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention_scores(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.size(-1)

        # (batch, num_heads, seq, d_k) x (batch, num_heads, d_k, seq) ===> (batch, num_heads, seq, seq)
        attention_scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -float('inf'))
        # (batch, num_heads, seq, seq) x (batch, num_heads, seq, seq) --> (batch, num_heads, seq, seq)
        attention_scores = nn.Softmax(dim=-1)(attention_scores)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch, num_heads, seq, seq) x (batch, num_heads, seq, d_k) ===> (batch, num_heads, seq, d_k)
        result = attention_scores @ value
        return result, attention_scores


    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, Seq_Len, d_model)
        key = self.w_k(k) # (Batch, Seq_Len, d_model)
        value = self.w_v(v) # (Batch, Seq_Len, d_model)

        B, T, C  = q.shape

        # (batch, seq, d_model) --> (batch, seq, num_heads, d_k) --> (batch, num_heads, seq, d_k)
        query = query.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention_scores(query, key, value, mask, dropout=self.dropout)
        # (batch, num_heads, d_model) 
        x = x.transpose(1, 2).contiguous().view(B, T, C)

        return self.w_o(x)
    

class LayerNormalization(nn.Module):
    def __init__(self, eps: int = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - x_mean)/(x_std + self.eps) + self.bias
    

class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.layer_norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, d_ff:int, num_heads: int, dropout:float):
        super().__init__()
        self.self_attention_block = MultiHeadAttentionBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.feed_forward_block = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, d_ff:int, num_heads: int, dropout:float):
        super().__init__()
        self.self_attention_block = MultiHeadAttentionBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.cross_attention_block = MultiHeadAttentionBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.feed_forward_block = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, src_mask, encoder_output, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, num_blocks: int, d_model: int, d_ff:int, num_heads: int, dropout:float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model, d_ff, num_heads, dropout) for _ in range(num_blocks)])
        self.layer_norm = LayerNormalization()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.layer_norm(x)
    
class Decoder(nn.Module):
    def __init__(self, num_blocks: int, d_model: int, d_ff:int, num_heads: int, dropout:float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model, d_ff, num_heads, dropout) for _ in range(num_blocks)])
        self.layer_norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, src_mask, encoder_output, tgt_mask)
        return self.layer_norm(x)
        
class ProjectionLayer(nn.Module):
    def __init__(self, vocab_size, d_model) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(x)
    
class Transformer(nn.Module):
    def __init__(self, num_encoder_blocks: int, num_decoder_blocks: int, src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int, d_ff:int, num_heads:int, dropout:float):
        super().__init__()
        self.encoder = Encoder(num_blocks=num_encoder_blocks, d_model=d_model,d_ff= d_ff, num_heads=num_heads, dropout=dropout)
        self.decoder = Decoder(num_blocks=num_decoder_blocks, d_model=d_model,d_ff= d_ff, num_heads=num_heads, dropout=dropout)
        self.src_embeddings = InputEmbeddings(vocab_size=src_vocab_size, d_model=d_model)
        self.src_position_embedings = PositionalEncoding(d_model=d_model, seq_len=src_seq_len, dropout=dropout)
        self.tgt_embeddings = InputEmbeddings(vocab_size=tgt_vocab_size, d_model=d_model)
        self.tgt_position_embedings = PositionalEncoding(d_model=d_model, seq_len=tgt_seq_len, dropout=dropout)
        self.proj_layer = ProjectionLayer(vocab_size=tgt_vocab_size, d_model=d_model)

    def encode(self, x, src_mask):
        x = self.src_embeddings(x)
        x = self.src_position_embedings(x)
        x = self.encoder(x, src_mask)
        return x

    def decode(self, x, src_mask, encoder_output, tgt_mask):
        x = self.tgt_embeddings(x)
        x = self.tgt_position_embedings(x)
        x = self.decoder(x, src_mask, encoder_output, tgt_mask)
        return x
    
    def project(self, x):
        return self.proj_layer(x)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        out = self.project(dec_out)
        return out
    

def build_transformer_model(src_vocab_size:int, tgt_vocab_size:int, src_seq_len: int, tgt_seq_len: int, N:int=4, H:int=8, d_model:int=512, d_ff:int=2048, dropout:float=0.1):
    transformer_model = Transformer(
        num_encoder_blocks=N,
        num_decoder_blocks=N,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_seq_len=src_seq_len,
        tgt_seq_len=tgt_seq_len,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=H,
        dropout=dropout
    )

    for p in transformer_model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)

    return transformer_model
