import torch
import torch.optim as optim
import torch.nn.functional as F

lr = 50
g = torch.Generator().manual_seed(42)
model = BigramClassifier()

optimizer = optim.SGD([model.W], lr=lr)
loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

