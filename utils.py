import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


def sharp_cosine_similarity(emb1, emb2, p=3.0):
  cos_sim = F.cosine_similarity(emb1, emb2)
  return torch.sign(cos_sim) * (torch.abs(cos_sim) + 1e-8) ** p

class ScoreModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = SentenceTransformer('sentence-transformers/sentence-t5-base')
    

  def forward(self, label, pred, p=3.0):
    if isinstance(pred, str): pred = [pred] * len(label)
    label_embed = self.model.encode(label)
    pred_embed = self.model.encode(pred)

    return sharp_cosine_similarity(torch.tensor(label_embed), torch.tensor(pred_embed), p).mean().item()