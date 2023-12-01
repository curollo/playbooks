import numpy as np
import torch
import pytorch_lightning as pl

def attention(queries, keys, values):
  d = queries.shape[-1]
  scores = torch.matmul(queries, keys.transpose(-2,-1))/np.sqrt(d)
  attention_weights = torch.nn.functional.softmax(scores, dim=-1)
  return torch.matmul(attention_weights, values)

class MultiHeadAttention(nn.Module):
  def __init__(self, embed_dim, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.embed_dim, self.num_heads = embed_dim, num_heads
    assert embed_dim % num_heads == 0
    self.projection_dim = embed_dim // num_heads

    self.W_q = nn.Linear(embed_dim, embed_dim)
    self.W_k = nn.Linear(embed_dim, embed_dim)
    self.W_v = nn.Linear(embed_dim, embed_dim)
    self.W_o = nn.Linear(embed_dim, embed_dim)

  def transpose(self, x):
    x = x.reshape(x.shape[0], x.shape[1], self.num_heads, self.projection_dim)
    return x.permute(0, 2, 1, 3)

  def transpose_output(self, x):
    x = x.permute(0, 2, 1, 3)
    return x.reshape(x.shape[0], x.shape[1], self.embed_dim)

  def forward(self, q, k, v):
    q = self.transpose(self.W_q(q))
    k = self.transpose(self.W_k(k))
    v = self.transpose(self.W_v(v))
    output = attention(q, k, v)
    return self.W_o(self.transpose_output(output))

class TransformerBro(nn.Module):
  def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
    super(TransformerBro, self).__init__()
    self.att = MultiHeadAttention(embed_dim, num_heads)
    self.ffn = nn.Sequential(
      nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim)
    )
    self.layernorm1 = nn.LayerNorm(embed_dim)
    self.layernorm2 = nn.LayerNorm(embed_dim)
    self.dropout = nn.Dropout(rate)

  def forward(self, x):
    x = self.layernorm1(x + self.dropout(self.att(x, x, x)))
    x = self.layernorm2(x + self.dropout(self.ffn(x)))
    return x

class TokenAndPositionEmbedding(nn.Module):
  def __init__(self, maxlen, vocab_size, embed_dim):
    super(TokenAndPositionEmbedding, self).__init__()
    self.token_emb = nn.Embedding(vocab_size, embed_dim)
    self.pos_emb = nn.Embedding(maxlen, embed_dim)
  def forward(self, x):
    pos = torch.arange(0, x.size(1), dtype=torch.int32, device=x.device)
    return self.token_emb(x) + self.pos_emb(pos).view(1, x.size(1), -1)



class SmolTransformer(pl.LightningModule):
  def __init__(self, seq_len=6, max_value=10, layer_count=2, embed_dim=128, num_heads=4, ff_dim=32):
    super().__init__()
    self.max_value = max_value
    self.model = nn.Sequential(
      TokenAndPositionEmbedding(seq_len, max_value, embed_dim),
      *[TransformerBlock(embed_dim, num_heads, ff_dim) for x in range(layer_count)],
      nn.Linear(embed_dim, max_value),
      nn.LogSoftmax(dim=-1))

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    x, y = batch
    output = self.model(x)
    loss = F.nll_loss(output.view(-1, self.max_value), y.view(-1))
    self.log("train_loss", loss)
    return loss

  def validation_step(self, val_batch, batch_idx):
    x, y = val_batch
    pred = self.model(x).argmax(dim=2)
    val_accuracy = (pred == y).type(torch.float).mean()
    self.log("val_accuracy", val_accuracy, prog_bar=True)
    torch.optim.Adam(self.parameters(), lr=3e-4)
