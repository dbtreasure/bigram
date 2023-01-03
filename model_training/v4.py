import torch
import torch.nn.functional as F
import numpy

epochs = 50
losses = []

def mini_batch(device, data_loader, stepn_fn):
    mini_batch_losses = []
    for x_batch, y_batch in data_loader:
        loss = stepn_fn(x_batch, y_batch)
        mini_batch_losses.append(loss)
    return numpy.mean(mini_batch_losses)

for epoch in range(epochs):
    loss = mini_batch(device, train_loader, train_step_fn)
    losses.append(loss)

    writer.add_scalars(main_tag=f'lr={lr}&momentum={momentum}&epochs={epochs}&batch={n_train_batch}&smoothing=0.01', tag_scalar_dict={'training': loss}, global_step=epoch)

writer.close()

checkpoint = {
    'epoch': epochs,
    'lr': lr,
    'momentum': momentum,
    'smoothing': '0.01+W**2.mean()',
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': losses
}

torch.save(checkpoint, 'checkpoint.pth')
# print last losses value
print(f'Final loss: {losses[-1]}')