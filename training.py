import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torchsummary import summary

from constants import SCRIPT_DIR
from network import Classifier


def MAV(values, L=10):
    # Moving average
    out = [values[0]]
    for v in values[1:]:
        out.append(out[-1] + (v - out[-1]) / L)
    return out


class Trainer:
    def __init__(
        self,
        train_dataset,
        valid_data,
        valid_labels,
        lr=2e-4,
        betas=(0.9, 0.999),
        eps=1e-16,
        depth=3,
        res_layers=3,
        kernel_size=4,
        dropout=0,
        inst_norm=True,
        base_channels=16,
        show_summary=False,
    ) -> None:
        self.train_dataset = train_dataset
        self.valid_data = valid_data
        self.valid_labels = valid_labels

        self.scaler = torch.amp.GradScaler("cuda")

        self.classifier = Classifier(
            depth=depth,
            res_layers=res_layers,
            base_channels=base_channels,
            dropout=dropout,
            inst_norm=inst_norm,
            kernel_size=kernel_size,
        ).cuda()

        if show_summary:
            summary(self.classifier, torch.ones(1, 3, 600, 600), depth=2)

        self.optim = torch.optim.Adam(
            self.classifier.parameters(), lr, betas=betas, eps=eps
        )
        self.loss_fn = nn.BCEWithLogitsLoss().cuda()

        self.loss_hist = []
        self.valid_hist = []
        self.lr_hist = [[0, lr]]
        self.best_score = None
        self.best_params = None

    def save_model(self):
        torch.save(
            {"class": self.classifier.state_dict(), "optim": self.optim.state_dict()},
            os.path.join(SCRIPT_DIR, "checkpoint.pt"),
        )

    def load_model(self):
        checkpoint = torch.load(os.path.join(SCRIPT_DIR, "checkpoint.pt"))
        self.classifier.load_state_dict(checkpoint["class"])
        self.optim.load_state_dict(checkpoint["optim"])

    def set_lr(self, lr):
        for g in self.optim.param_groups:
            g["lr"] = lr
        idx = len(self.loss_hist) - 1
        self.lr_hist.append([idx, self.lr_hist[-1][-1]])
        self.lr_hist.append([idx, lr])

    def train(self, epochs=1000, valid_ival=10, lr_ival=300, gamma=0.7):
        for i in tqdm(range(epochs)):
            self.step()
            if i > 0 and i % lr_ival == 0:
                self.set_lr(self.lr_hist[-1][-1] * gamma)
            if i % valid_ival == 0:
                self.eval()
        self.eval()
        print(f"Final Loss: {MAV(self.loss_hist)[-1]:.2e}")

    def eval(self, use_best_params=False):
        if use_best_params:
            self.classifier.load_state_dict(self.best_params)

        # Evaluate prediction accuracy
        with torch.no_grad():
            self.classifier.eval()
            with torch.autocast(device_type="cuda"):
                pred = self.classifier(self.valid_data)
                # conv_pred = np.argmax(pred.cpu().numpy(), -1)
                conv_pred = pred.cpu().flatten().numpy() > 0.5
            score = f1_score(self.valid_labels, conv_pred)
            self.classifier.train()

        self.valid_hist.append([len(self.loss_hist) - 1, score])
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_params = self.classifier.state_dict()
        return conv_pred

    def step(self):
        # Perform one training iteration on one batch
        self.optim.zero_grad()

        data, labels = self.train_dataset.make_batch()

        with torch.autocast(device_type="cuda"):
            pred = self.classifier(data)
        loss = self.loss_fn(pred, labels)

        self.loss_hist.append(loss.item())

        # loss.backward()
        # self.optim.step()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optim)
        self.scaler.update()

    def plot_hist(
        self, title=None, width=8, height=4, avlen=10, alpha=0.3, show_CM=False
    ):
        fig, ax = plt.subplots(1, 2 if show_CM else 1)
        fig.set_size_inches(width, height)
        axs = fig.axes

        X, Y = list(zip(*self.valid_hist))
        axs[0].plot(
            MAV(self.loss_hist, avlen * 3),
            "b-",
            path_effects=[pe.Stroke(linewidth=3, foreground="k"), pe.Normal()],
        )
        axs[0].plot(
            X,
            MAV(Y, avlen),
            "r-",
            path_effects=[pe.Stroke(linewidth=3, foreground="k"), pe.Normal()],
        )
        axs[0].plot(0, 0, "g.-")
        axs[0].plot(self.loss_hist, "b.", alpha=alpha, zorder=-10)
        axs[0].plot(X, Y, "r.", alpha=alpha, zorder=-10)
        axs[0].set_xlabel("Batches")
        axs[0].set_ylabel("Metric")
        axs[0].grid()
        axs[0].legend(["Training Loss", "Valid. F1-Score", "Learning Rate"])
        if title is not None:
            axs[0].set_title(title)
        axs[0].hlines([0, 1], X[0], X[-1], "k", "dashed")
        axs[0].set_ylim(-0.05, 1.05)

        ax1 = axs[0].twinx()
        X, Y = list(
            zip(*(self.lr_hist + [[len(self.loss_hist) - 1, self.lr_hist[-1][-1]]]))
        )
        ax1.plot(X, Y, "g.-")
        ax1.set_ylabel("Learning Rate")
        ax1.set_yscale("log")

        if show_CM:
            pred = self.eval(use_best_params=True)
            cm = confusion_matrix(self.valid_labels, pred)
            # score = f1_score(self.valid_labels, pred)
            dis = ConfusionMatrixDisplay(cm, display_labels=["Siirt", "Kirmizi"])
            dis.plot(ax=axs[1])
            axs[1].set_title(f"Confusion Matrix")
            # axs[1].set_title(f"Confusion Matrix (F1 Score: {score:.3f})")

        plt.show()

    def plot_confusion_matrix(self, width=6, height=4):
        pred = self.eval(use_best_params=True)
        cm = confusion_matrix(self.valid_labels, pred)
        # score = f1_score(self.valid_labels, pred)
        dis = ConfusionMatrixDisplay(cm, display_labels=["Siirt", "Kirmizi"])
        dis.plot()
        plt.gca().set_title(f"Confusion Matrix")
        plt.gcf().set_size_inches(width, height)
        plt.show()
