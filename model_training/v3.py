import torch
import torch.nn.functional as F
import numpy

epochs = 500
losses = []
val_losses = []

def mini_batch(device, data_loader, stepn_fn):
    mini_batch_losses = []
    for x_batch, y_batch in data_loader:
        loss = stepn_fn(x_batch, y_batch)
        mini_batch_losses.append(loss)
    return numpy.mean(mini_batch_losses)

for epoch in range(epochs):
    # print a statement that says `epoch: <epoch number>`
    print(f'epoch: {epoch}')
    loss = mini_batch(device, train_loader, train_step_fn)
    losses.append(loss)

    val_loss = mini_batch(device, validation_loader, validation_step_fn)
    val_losses.append(val_loss)

