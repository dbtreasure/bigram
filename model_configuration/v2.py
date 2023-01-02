import torch
import torch.optim as optim

def make_train_step_fn(model, loss_fn, optimizer, smoothing_count):
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(yhat, y) + pow(1, -smoothing_count)
        # create a variable that is 1 to the power of minus smoothing count

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
    return train_step

# make a validation step function
def make_validation_step_fn(model, loss_fn, smoothing_count):
    def validation_step(x, y):
        model.eval()
        # NO GRADIENTS IN VALIDATION
        with torch.no_grad():
            yhat = model(x)
            loss = loss_fn(yhat, y) + pow(1, -smoothing_count)
            return loss.item()
    return validation_step

# make a test step function
def make_test_step_fn(model, loss_fn, smoothing_count):
    def test_step(x, y):
        model.eval()
        # NO GRADIENTS IN VALIDATION
        with torch.no_grad():
            yhat = model(x)
            loss = loss_fn(yhat, y) + pow(1, -smoothing_count)
            return loss.item()
    return test_step

# create a higher order function to wrap make_train_step_fn for providing a smoothing count
def make_train_step_fn_with_smoothing_count(smoothing_count):
    def train_step_fn():
        return make_train_step_fn(model, loss_fn, optimizer, smoothing_count)
    return train_step_fn

def make_valid_step_fn_with_smoothing_count(smoothing_count):
    def train_step_fn():
        return make_validation_step_fn(model, loss_fn, smoothing_count)
    return train_step_fn

def make_test_step_fn_with_smoothing_count(smoothing_count):
    def train_step_fn():
        return make_test_step_fn(model, loss_fn, smoothing_count)
    return train_step_fn

lr = 50
g = torch.Generator().manual_seed(42)
model = BigramClassifier()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()

# train_step_fn = make_train_step_fn(model, loss_fn, optimizer, smoothing_count)
# validation_step_fn = make_validation_step_fn(model, loss_fn, 1)
# test_step_fn = make_test_step_fn(model, loss_fn, 1)

writer = SummaryWriter('runs/bigram_classifier')
x_dummy, y_dummy = next(iter(train_loader))
writer.add_graph(model, x_dummy.to(device))