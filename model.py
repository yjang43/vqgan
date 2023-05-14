import torch
import torch.nn as nn
import torch.nn.functional as F
from module import Encoder, Decoder
from function import VectorQuantization
from einops import rearrange


class VQVAE(nn.Module):

    def __init__(self, in_dim=3, hid_dim=256, k=1000, beta=0.25):

        super().__init__()
        self.E = Encoder(in_dim, hid_dim)
        self.D = Decoder(hid_dim, in_dim)
        self.vq = VectorQuantization.apply
        self.codebook = nn.Parameter(torch.rand(k, hid_dim))

        self.beta = beta
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)

    def forward(self, x):

        z = self.E(x)
        b, c, h, w = z.shape
        z = rearrange(z, "b c h w -> (b h w) c")

        # will be used for vq_loss and commitment_loss
        e = self.vq(z.detach(), self.codebook)

        # will be used for reconstruction_loss
        z_q = self.vq(z, self.codebook.detach())
        z_q = rearrange(z_q, "(b h w) c -> b c h w", b=b, h=h, w=w, c=c)

        x_hat = self.D(z_q)
        x_hat = F.tanh(x_hat)
        return x_hat, z, e

    def optimize(self, x, x_hat, z, e):
        
        self.optimizer.zero_grad()

        reconstruction_loss = F.mse_loss(x_hat, x)
        vq_loss = F.mse_loss(z.detach(), e)
        commitment_loss = F.mse_loss(z, e.detach())

        total_loss = reconstruction_loss + vq_loss + self.beta*commitment_loss

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), (
            reconstruction_loss.item(),
            vq_loss.item(),
            commitment_loss.item()
        )

    @torch.no_grad()
    def inference(self, x):
        
        was_train = False
        if self.training:
            was_train = True
            self.eval()

        z = self.E(x)
        b, c, h, w = z.shape
        z = rearrange(z, "b c h w -> (b h w) c")
        z_q = self.vq(z, self.codebook)
        z_q = rearrange(z_q, "(b h w) c -> b c h w", b=b, h=h, w=w, c=c)

        x_hat = self.D(z_q)
        x_hat = F.tanh(x_hat)

        if was_train:
            self.train()

        return x_hat