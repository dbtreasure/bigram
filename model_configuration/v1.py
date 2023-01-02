import torch
import torch.optim as optim


def make_train_step_fn(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(yhat, y) + 0.01 * model.W.pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step

lr = 50
g = torch.Generator().manual_seed(42)
model = BigramClassifier()

optimizer = optim.SGD([model.W], lr=lr, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
