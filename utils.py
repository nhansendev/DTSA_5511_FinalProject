import torch
import numpy as np
import matplotlib.pyplot as plt


def sum_model_params(model):
    return sum(p.numel() for p in model.parameters())


def plot_img_tensor(data, title=None, r=1, c=6, w=12, h=2):
    if isinstance(data, np.ndarray):
        imgs = data
    else:
        tmp = torch.permute(data[: r * c], (0, 2, 3, 1))
        imgs = ((tmp.cpu().detach().numpy() + 1) * 127.5).astype(int)

    fig, ax = plt.subplots(r, c)
    axs = fig.axes
    fig.set_size_inches(w, h)

    for i, a in enumerate(axs):
        a.imshow(imgs[i])
        a.axis("off")

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()
