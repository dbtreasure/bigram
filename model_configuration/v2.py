import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

def smoothing():
    return + 0.01*(model.W**2).mean()

def make_train_step_fn(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(yhat, y) + smoothing()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
    return train_step

# make a validation step function
def make_validation_step_fn(model, loss_fn):
    def validation_step(x, y):
        model.eval()
        # NO GRADIENTS IN VALIDATION
        with torch.no_grad():
            yhat = model(x)
            loss = loss_fn(yhat, y)
            return loss.item()
    return validation_step

lr = 50
momentum = 0.9
model = BigramClassifier()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
loss_fn = torch.nn.CrossEntropyLoss()

train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
validation_step_fn = make_validation_step_fn(model, loss_fn)

writer = SummaryWriter('runs/bigram_classifier')
x_dummy, y_dummy = next(iter(train_loader))
writer.add_graph(model, x_dummy.to(device))