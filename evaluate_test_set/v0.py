test_loss = mini_batch(device, test_loader, validation_step_fn)
print(f'Test loss: {test_loss}')