epochs = 500
losses = []

for epoch in range(epochs):
    loss = train_step_fn(xs_train_encoded, ys_train)
    losses.append(loss)
