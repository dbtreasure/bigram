import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

def smoothing():
    return + 0.01*(trigram_model.W**2).mean()

def make_trigram_train_step_fn(model, loss_fn, optimizer):
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
def make_trigram_validation_step_fn(model, loss_fn):
    def validation_step(x, y):
        model.eval()
        # NO GRADIENTS IN VALIDATION
        with torch.no_grad():
            yhat = model(x)
            loss = loss_fn(yhat, y)
            return loss.item()
    return validation_step

trigram_lr = 50
trigram_momentum = 0.9
trigram_model = TrigramClassifier()

trigram_optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
trigram_loss_fn = torch.nn.CrossEntropyLoss()

trigram_train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
trigram_validation_step_fn = make_validation_step_fn(model, loss_fn)

trigram_writer = SummaryWriter('runs/trigram_classifier')
trigram_x_dummy, trigram_y_dummy = next(iter(trigram_train_loader))
writer.add_graph(trigram_model, trigram_x_dummy.to(device))