import torch
import torch.nn as nn

preds = torch.rand((2, 3, 4), dtype=torch.float)
print(preds.shape[-1])
preds = preds.view(-1, preds.size(-1))
print(preds.size())
