from tqdm import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import torch
from iou import iou


def run(model, dataloader, loss, optimizer=None, device='cpu'):

    loop = tqdm(dataloader, leave=True)
    mean_loss = []

    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        current_loss = loss(pred, y)
        mean_loss.append(current_loss.item())
        total_loss += current_loss.item()
        if optimizer is not None:
            current_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        loop.set_postfix(loss=current_loss.item())

    return total_loss / len(mean_loss)


# функции вывода графиков
def show_losses(train_loss_hist, test_loss_hist=None):
    clear_output()

    plt.figure(figsize=(12,4))

    plt.subplot(1, 2, 1)
    plt.title('Train Loss')
    plt.plot(np.arange(len(train_loss_hist)), train_loss_hist)
    plt.yscale('log')
    plt.grid()

    if test_loss_hist is not None:
        plt.subplot(1, 2, 2)
        plt.title('Test Loss')
        plt.plot(np.arange(len(test_loss_hist)), test_loss_hist)
        plt.yscale('log')
        plt.grid()

    plt.show()

