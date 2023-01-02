import torch
import torch.optim as optim

def make_train_step_fn(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(yhat, y) # + .001 * model.W.pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step

# make a validation step function
def make_validation_step_fn(model, loss_fn):
    def validation_step(x, y):
        model.eval()
        # NO GRADIENTS IN VALIDATION
        with torch.no_grad():
            yhat = model(x)
            loss = loss_fn(yhat, y) # + .001 * model.W.pow(2).mean()
            return loss.item()
    return validation_step

# make a test step function
def make_test_step_fn(model, loss_fn):
    def test_step(x, y):
        model.eval()
        # NO GRADIENTS IN VALIDATION
        with torch.no_grad():
            yhat = model(x)
            loss = loss_fn(yhat, y) # + .001 * model.W.pow(2).mean()
            return loss.item()
    return test_step


lr = 50
g = torch.Generator().manual_seed(42)
model = BigramClassifier()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()

train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
validation_step_fn = make_validation_step_fn(model, loss_fn)
test_step_fn = make_test_step_fn(model, loss_fn)

writer = SummaryWriter('runs/bigram_classifier')
x_dummy, y_dummy = next(iter(train_loader))
writer.add_graph(model, x_dummy.to(device))