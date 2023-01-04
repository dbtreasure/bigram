import torch
import torch.nn.functional as F
import numpy

trigram_epochs = 10
trigram_losses = []

def trigram_mini_batch(device, data_loader, stepn_fn):
    mini_batch_losses = []
    for x_batch, y_batch in data_loader:
        loss = stepn_fn(x_batch, y_batch)
        mini_batch_losses.append(loss)
    return numpy.mean(mini_batch_losses)

for epoch in range(trigram_epochs):
    loss = trigram_mini_batch(device, train_loader, train_step_fn)
    trigram_losses.append(loss)

    trigram_writer.add_scalars(main_tag=f'TRIGRAM&lr={trigram_lr}&momentum={trigram_momentum}&epochs={trigram_epochs}&batch={trigram_n_train_batch}&smoothing=0.01', tag_scalar_dict={'training': loss}, global_step=epoch)

writer.close()

checkpoint = {
    'epoch': trigram_epochs,
    'lr': trigram_lr,
    'momentum': trigram_momentum,
    'smoothing': '0.01+W**2.mean()',
    'model_state_dict': trigram_model.state_dict(),
    'optimizer_state_dict': trigram_optimizer.state_dict(),
    'loss': trigram_losses
}

torch.save(checkpoint, 'trigram_checkpoint.pth')
# print last losses value
print(f'Final trigram training loss: {trigram_losses[-1]}')