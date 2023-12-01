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
