import torch
import torch.nn.functional as F
import numpy

epochs = 20
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

    # create a for in loop of range of 1 to 5
    for i in range(10, 16):
        print(f'epoch: {epoch}, smoothing_count: {i}')
        step_fn = make_train_step_fn_with_smoothing_count(i)
        loss = mini_batch(device, train_loader, step_fn())
        losses.append(loss)

        val_step_fn = make_valid_step_fn_with_smoothing_count(i)
        val_loss = mini_batch(device, validation_loader, val_step_fn())
        val_losses.append(val_loss)

        writer.add_scalars(main_tag=f'loss+smoothing_i=1e-{i}', tag_scalar_dict={'training': loss, 'validation': val_loss}, global_step=epoch)

writer.close()