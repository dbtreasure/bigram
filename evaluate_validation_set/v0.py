# evaluate the model on the validation set

val_loss = mini_batch(device, validation_loader, validation_step_fn)
print(val_loss)
