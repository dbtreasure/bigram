val_loss = mini_batch(device, validation_loader, validation_step_fn)
print(f'Validation loss: {val_loss}')
