import torch
import numpy as np
import matplotlib.pyplot as plt


def sum_model_params(model):
    out = 0
    for p in model.parameters():
        try:
            out += p.numel()
        except ValueError:
            pass
    return out


def plot_img_tensor(data, title=None, r=1, c=6, w=12, h=2, labels=None):
    if isinstance(data, np.ndarray):
        imgs = data
    else:
        tmp = torch.permute(data[: r * c], (0, 2, 3, 1))
        imgs = ((tmp.cpu().detach().numpy() + 1) * 127.5).astype(int)

    fig, ax = plt.subplots(r, c)
    axs = fig.axes
    fig.set_size_inches(w, h)

    for i, a in enumerate(axs):
        try:
            if labels is not None:
                try:
                    a.set_title(labels[i])
                except IndexError:
                    a.set_title("-")
            a.imshow(imgs[i])
            a.axis("off")
        except IndexError:
            a.axis("off")

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()


def plot_features(
    model, input, layers=["0", "1"], num_per_layer=8, s=1.5, title=None, verbose=False
):
    # Model assumes a nn.Sequential() instance
    r = len(layers)
    c = num_per_layer
    w = s * c
    h = s * r

    if len(input.shape) == 3:
        input = input.unsqueeze(0)
    assert input.shape[0] == 1, "Use batch of size one"

    outputs = []
    with torch.no_grad():
        for k, v in model._modules.items():
            input = v(input)
            if verbose:
                print(f"Output {k}: {input.shape}")
            if k in layers:
                outputs.append([k, input])
            if len(outputs) == len(layers):
                break

    fig, ax = plt.subplots(r, c)
    axs = fig.axes
    fig.set_size_inches(w, h)

    if title is not None:
        plt.suptitle(title)

    for j, (k, val) in enumerate(outputs):
        if len(val.shape) < 4:
            break
        val = torch.permute(val.squeeze(0), (1, 2, 0)).cpu().numpy()
        if verbose:
            print(f"Resulting shape [width, height, channels]: {val.shape}")
        for i in range(num_per_layer):
            idx = j * c + i
            try:
                axs[idx].imshow(val[:, :, i])
            except IndexError:
                pass
            if i == 0:
                axs[idx].set_ylabel(k)

            axs[idx].set_xticks([])
            axs[idx].set_yticks([])

    plt.tight_layout()

    plt.show()
