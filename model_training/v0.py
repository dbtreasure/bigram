epochs = 500

for epoch in range(epochs):
    model.train()
    logits = model(xs_train_encoded)

    # compute the loss using pytorch crossentropyloss
    loss = loss_fn(logits, ys_train) + 0.01 * model.W.pow(2).mean()
    print(loss.item())
    loss.backward()

    optimizer.step()

    optimizer.zero_grad()