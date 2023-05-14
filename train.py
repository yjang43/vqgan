import os
import logging
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
from tqdm import tqdm
from model import VQVAE
from util import compute_grad, unnormalize


# args
device = "cuda"
num_epoch = 50
result_dir = "./result"


if __name__ == "__main__":

    os.makedirs(result_dir, exist_ok=True)

    # make log file
    os.makedirs("./log", exist_ok=True)
    cur_time_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    logging.basicConfig(
        filename=f"./log/{cur_time_str}.log",
        filemode="w",
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
        force=True
    )


    # prepare dataset
    mean = torch.Tensor((0.5, 0.5, 0.5))
    std = torch.Tensor((1.0, 1.0, 1.0))
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    dataset = CIFAR10("./cifar10/", download=True, transform=preprocess, train=True)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    test_dataset = CIFAR10("./cifar10/", download=True, transform=preprocess, train=False)
    test_batch, _ = next(iter(DataLoader(test_dataset, batch_size=16)))
    test_batch = test_batch.to(device)


    # initialize model
    model = VQVAE()
    model.to(device)
    model.train()

    num_iter = num_epoch * len(dataloader)
    prog_bar = tqdm(range(num_iter))


    for epoch in range(num_epoch):

        for x, _ in dataloader:

            x = x.to(device)
            x_hat, z, e = model(x)
            loss, (rec_loss, vq_loss, com_loss) = model.optimize(x, x_hat, z, e)

            prog_bar.set_description(f"loss: {round(loss, 3)}")
            prog_bar.update()

            if (prog_bar.n) % 100 == 0:
                logging.info(
                    f"loss: {loss} | "
                    f"reconstruction loss: {rec_loss}, "
                    f"vq loss: {vq_loss}, "
                    f"commitment loss: {com_loss}"
                )
                logging.info(f"grad norm: {compute_grad(model.parameters())}")

                test_batch_pred = model.inference(test_batch)
                fig, axs = plt.subplots(2, 1)
                img = make_grid(unnormalize(test_batch.cpu(), mean, std), nrow=4)
                pred_img = make_grid(unnormalize(test_batch_pred.cpu(), mean, std), nrow=4)
                axs[0].imshow((img.permute(1, 2, 0).numpy()))
                axs[0].set_title("true x")
                axs[1].imshow(pred_img.permute(1, 2, 0).numpy())
                axs[1].set_title("predicted x")
                fig.savefig(f"./{result_dir}/{prog_bar.n}.png")
                plt.close(fig)
                