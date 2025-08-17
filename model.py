import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape = (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div =  torch.pow(10000, torch.arange(0, d_model, 2)/d_model).float()
        # Apply sin to even positions
        pe[:, 0::2] = torch.sin(position/div)
        # Apply cos to odd positions
        pe[:, 1::2] = torch.cos(position/div)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    
    def forward(self, x):
        x = x + (self.pe[:, :x.size(1), :].requires_grad_(False))
        return self.dropout(x)

 
class LayerNormalization(nn.Module):
    def __init__(self, eps: int = -1e6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - x_mean) / (x_std + self.eps) + self.bias
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        layers = [
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        ]
        self.ff = nn.Sequential(*layers)

    def forward(self, x):
        return self.ff(x)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_v = d_model//h

        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_v = nn.Linear(d_model, d_model) # Wv
        self.w_k = nn.Linear(d_model, d_model) # Wk

        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.drop_out = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]

        # (Batch, h, Seq_Len, d_v) x (Batch, h, d_v, Seq_Len) => (Batch, h, Seq_Len, Seq_Len) 
        attention_scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = nn.Softmax(dim=-1)(attention_scores)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # (Batch, h, Seq_Len, Seq_Len) * (Batch, h, Seq_Len, d_k) =>  (Batch, h, Seq_Len, d_k)
        result = attention_scores @ value
        return result, attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (Batch, Seq_Len, d_model) => (Batch, Seq_Len, h, d_v) => (Batch, h, Seq_Len, d_v)
        query = query.view(query.size(0), query.size(1), self.h, self.d_v).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), self.h, self.d_v).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.h, self.d_v).transpose(1, 2)

        # x = (Batch, h, Seq_Len, d_k), self.attention_scores = (Batch, h, Seq_Len, Seq_Len) 
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.drop_out)

        # d_v == d_k
        # (Batch, h, Seq_Len, d_v) --> (Batch, Seq_Len, h, d_v) --> (Batch, Seq_Len, h*d_v)
        x = x.transpose(1, 2).reshape(x.shape[0], -1, self.h*self.d_v)
        
        # (Batch, Seq_Len, h*d_v) --> (Batch, Seq_Len, d_model) Remember: h*d_v == d_model
        return self.w_o(x)
    

class ResidualConnection(nn.Module):
    def __init__(self, dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) 
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock, cross_attention_block:MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return nn.Softmax(dim=-1)(self.proj(x))
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder:Decoder, src_embeddings: InputEmbeddings, tgt_embeddings: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embeddings = src_embeddings
        self.tgt_embeddings = tgt_embeddings
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, x, src_mask):
        x = self.src_embeddings(x)
        x = self.src_pos(x)
        x = self.encoder(x, src_mask)
        return x

    def decode(self, encoder_output, src_mask, x, tgt_mask):
        x = self.tgt_embeddings(x)
        x = self.tgt_pos(x)
        x = self.decoder(x, encoder_output, src_mask, tgt_mask)
        return x
        
    def project(self, x):
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: int = 0.1, d_ff: int = 2048):

    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(self_attention_block, cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    ## intialize the parameters with Xe
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        
    return transformer

