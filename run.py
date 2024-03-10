import torch
from torch import device


def run(model, dataloader, loss, optimizer=None, device='cpu'):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = loss

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        total_loss += loss(pred, y).item()
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / len()