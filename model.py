from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.optim.lr_scheduler import _LRScheduler
from utils import Literal

from torchvision.datasets.mnist import MNIST
from torchvision import transforms


Representation = Literal['pool', 'tower', 'pyramid']


class Conv2dLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2dLSTMCell, self).__init__()

        kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding)

        in_channels += out_channels

        self.forget = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.input = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.output = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.state = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, input, states):
        (cell, hidden) = states
        input = torch.cat((hidden, input), dim=1)

        forget_gate = torch.sigmoid(self.forget(input))
        input_gate = torch.sigmoid(self.input(input))
        output_gate = torch.sigmoid(self.output(input))
        state_gate = torch.tanh(self.state(input))

        # Update internal cell state
        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * torch.tanh(cell)

        return cell, hidden


class InferenceCore(nn.Module):
    def __init__(self):
        super(InferenceCore, self).__init__()
        self.downsample_x = nn.Conv2d(3, 3, kernel_size=4, stride=4, padding=0, bias=False)
        self.upsample_v = nn.ConvTranspose2d(7, 7, kernel_size=16, stride=16, padding=0, bias=False)
        self.upsample_r = nn.ConvTranspose2d(256, 256, kernel_size=16, stride=16, padding=0, bias=False)
        self.downsample_u = nn.Conv2d(128, 128, kernel_size=4, stride=4, padding=0, bias=False)
        self.core = Conv2dLSTMCell(3+7+256+2*128, 128, kernel_size=5, stride=1, padding=2)

    def forward(self, x, v, r, c_e, h_e, h_g, u):
        x = self.downsample_x(x)
        v = self.upsample_v(v.view(-1, 7, 1, 1))
        if r.size(2) != h_e.size(2):
            r = self.upsample_r(r)
        u = self.downsample_u(u)
        c_e, h_e = self.core(torch.cat((x, v, r, h_g, u), dim=1), (c_e, h_e))

        return c_e, h_e


class GenerationCore(nn.Module):
    def __init__(self):
        super(GenerationCore, self).__init__()
        self.upsample_v = nn.ConvTranspose2d(7, 7, kernel_size=16, stride=16, padding=0, bias=False)
        self.upsample_r = nn.ConvTranspose2d(256, 256, kernel_size=16, stride=16, padding=0, bias=False)
        self.core = Conv2dLSTMCell(7+256+3, 128, kernel_size=5, stride=1, padding=2)
        self.upsample_h = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4, padding=0, bias=False)

    def forward(self, v, r, c_g, h_g, u, z):
        v = self.upsample_v(v.view(-1, 7, 1, 1))
        if r.size(2) != h_g.size(2):
            r = self.upsample_r(r)
        c_g, h_g = self.core(torch.cat((v, r, z), dim=1), (c_g, h_g))
        u = self.upsample_h(h_g) + u

        return c_g, h_g, u


class Pyramid(nn.Module):
    def __init__(self):
        super(Pyramid, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(7+3, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=8, stride=8),
            nn.ReLU()
        )

    def forward(self, x, v):
        # Broadcast
        v = v.view(-1, 7, 1, 1).repeat(1, 1, 64, 64)
        r = self.net(torch.cat((v, x), dim=1))

        return r


class Tower(nn.Module):
    def __init__(self):
        super(Tower, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256+7, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256+7, 128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=1, stride=1)

    def forward(self, x, v):
        # Resisual connection
        skip_in = F.relu(self.conv1(x))
        skip_out = F.relu(self.conv2(skip_in))

        r = F.relu(self.conv3(skip_in))
        r = F.relu(self.conv4(r)) + skip_out

        # Broadcast
        v = v.view(v.size(0), 7, 1, 1).repeat(1, 1, 16, 16)

        # Resisual connection
        # Concatenate
        skip_in = torch.cat((r, v), dim=1)
        skip_out = F.relu(self.conv5(skip_in))

        r = F.relu(self.conv6(skip_in))
        r = F.relu(self.conv7(r)) + skip_out
        r = F.relu(self.conv8(r))

        return r


class Pool(nn.Module):
    def __init__(self):
        super(Pool, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256+7, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256+7, 128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.pool = nn.AvgPool2d(16)

    def forward(self, x, v):
        # Resisual connection
        skip_in = F.relu(self.conv1(x))
        skip_out = F.relu(self.conv2(skip_in))

        r = F.relu(self.conv3(skip_in))
        r = F.relu(self.conv4(r)) + skip_out

        # Broadcast
        v = v.view(v.size(0), 7, 1, 1).repeat(1, 1, 16, 16)

        # Resisual connection
        # Concatenate
        skip_in = torch.cat((r, v), dim=1)
        skip_out = F.relu(self.conv5(skip_in))

        r = F.relu(self.conv6(skip_in))
        r = F.relu(self.conv7(r)) + skip_out
        r = F.relu(self.conv8(r))

        # Pool
        r = self.pool(r)

        return r


class GQN(nn.Module):
    def __init__(self, representation="pool", L=12, shared_core: bool = False):
        super(GQN, self).__init__()

        # Number of generative layers
        self.L = L

        # Representation network
        self.representation = representation
        if representation == "pyramid":
            self.phi = Pyramid()
        elif representation == "tower":
            self.phi = Tower()
        elif representation == "pool":
            self.phi = Pool()

        # Generation network
        self.shared_core = shared_core
        if shared_core:
            self.inference_core = InferenceCore()
            self.generation_core = GenerationCore()
        else:
            self.inference_core = nn.ModuleList([InferenceCore() for _ in range(L)])
            self.generation_core = nn.ModuleList([GenerationCore() for _ in range(L)])

        self.eta_pi = nn.Conv2d(128, 2*3, kernel_size=5, stride=1, padding=2)
        self.eta_g = nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0)
        self.eta_e = nn.Conv2d(128, 2*3, kernel_size=5, stride=1, padding=2)

    # EstimateELBO
    def forward(self, x, v, v_q, x_q, sigma):
        B, M, *_ = x.size()

        # Scene encoder
        if self.representation == 'tower':
            r = x.new_zeros((B, 256, 16, 16))
        else:
            r = x.new_zeros((B, 256, 1, 1))
        for k in range(M):
            r_k = self.phi(x[:, k], v[:, k])
            r += r_k

        # Generator initial state
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))

        # Inference initial state
        c_e = x.new_zeros((B, 128, 16, 16))
        h_e = x.new_zeros((B, 128, 16, 16))

        elbo = 0
        for l in range(self.L):
            # Prior factor
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_g), 3, dim=1)
            std_pi = torch.exp(0.5*logvar_pi)
            pi = Normal(mu_pi, std_pi)

            # Inference state update
            if self.shared_core:
                c_e, h_e = self.inference_core(x_q, v_q, r, c_e, h_e, h_g, u)
            else:
                c_e, h_e = self.inference_core[l](x_q, v_q, r, c_e, h_e, h_g, u)

            # Posterior factor
            mu_q, logvar_q = torch.split(self.eta_e(h_e), 3, dim=1)
            std_q = torch.exp(0.5*logvar_q)
            q = Normal(mu_q, std_q)

            # Posterior sample
            z = q.rsample()

            # Generator state update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)

            # ELBO KL contribution update
            elbo -= torch.sum(kl_divergence(q, pi), dim=[1, 2, 3])

        # ELBO likelihood contribution update
        elbo += torch.sum(Normal(self.eta_g(u), sigma).log_prob(x_q), dim=[1, 2, 3])

        return elbo

    def generate(self, x, v, v_q):
        B, M, *_ = x.size()

        # Scene encoder
        if self.representation == 'tower':
            r = x.new_zeros((B, 256, 16, 16))
        else:
            r = x.new_zeros((B, 256, 1, 1))
        for k in range(M):
            r_k = self.phi(x[:, k], v[:, k])
            r += r_k

        # Initial state
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))

        for l in range(self.L):
            # Prior factor
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_g), 3, dim=1)
            std_pi = torch.exp(0.5*logvar_pi)
            pi = Normal(mu_pi, std_pi)

            # Prior sample
            z = pi.sample()

            # State update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)

        # Image sample
        mu = self.eta_g(u)

        return torch.clamp(mu, 0, 1)

    def kl_divergence(self, x, v, v_q, x_q):
        B, M, *_ = x.size()

        # Scene encoder
        if self.representation == 'tower':
            r = x.new_zeros((B, 256, 16, 16))
        else:
            r = x.new_zeros((B, 256, 1, 1))
        for k in range(M):
            r_k = self.phi(x[:, k], v[:, k])
            r += r_k

        # Generator initial state
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))

        # Inference initial state
        c_e = x.new_zeros((B, 128, 16, 16))
        h_e = x.new_zeros((B, 128, 16, 16))

        kl = 0
        for l in range(self.L):
            # Prior factor
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_g), 3, dim=1)
            std_pi = torch.exp(0.5*logvar_pi)
            pi = Normal(mu_pi, std_pi)

            # Inference state update
            if self.shared_core:
                c_e, h_e = self.inference_core(x_q, v_q, r, c_e, h_e, h_g, u)
            else:
                c_e, h_e = self.inference_core[l](x_q, v_q, r, c_e, h_e, h_g, u)

            # Posterior factor
            mu_q, logvar_q = torch.split(self.eta_e(h_e), 3, dim=1)
            std_q = torch.exp(0.5*logvar_q)
            q = Normal(mu_q, std_q)

            # Posterior sample
            z = q.rsample()

            # Generator state update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)

            # ELBO KL contribution update
            kl += torch.sum(kl_divergence(q, pi), dim=[1, 2, 3])

        return kl

    def reconstruct(self, x, v, v_q, x_q):
        B, M, *_ = x.size()

        # Scene encoder
        if self.representation == 'tower':
            r = x.new_zeros((B, 256, 16, 16))
        else:
            r = x.new_zeros((B, 256, 1, 1))
        for k in range(M):
            r_k = self.phi(x[:, k], v[:, k])
            r += r_k

        # Generator initial state
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))

        # Inference initial state
        c_e = x.new_zeros((B, 128, 16, 16))
        h_e = x.new_zeros((B, 128, 16, 16))

        for l in range(self.L):
            # Inference state update
            if self.shared_core:
                c_e, h_e = self.inference_core(x_q, v_q, r, c_e, h_e, h_g, u)
            else:
                c_e, h_e = self.inference_core[l](x_q, v_q, r, c_e, h_e, h_g, u)

            # Posterior factor
            mu_q, logvar_q = torch.split(self.eta_e(h_e), 3, dim=1)
            std_q = torch.exp(0.5*logvar_q)
            q = Normal(mu_q, std_q)

            # Posterior sample
            z = q.rsample()

            # Generator state update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)

        mu = self.eta_g(u)

        return torch.clamp(mu, 0, 1)


class AnnealingStepLR(_LRScheduler):
    def __init__(self, optimizer, mu_i=5e-4, mu_f=5e-5, n=1.6e6):
        self.mu_i = mu_i
        self.mu_f = mu_f
        self.n = n
        super(AnnealingStepLR, self).__init__(optimizer)

    def get_lr(self):
        return [max(self.mu_f + (self.mu_i - self.mu_f) * (1.0 - self.last_epoch / self.n), self.mu_f) for base_lr in self.base_lrs]


class GQNModel(pl.LightningModule):
    def __init__(self, representation: Representation = 'pool', shared_core: bool = False, layers: int = 12):
        super().__init__()
        self.save_hyperparameters()
        self.gqn = GQN(representation=representation, shared_core=shared_core, L=layers)
        self.sigma_i, self.sigma_f = 2.0, 0.7
        self.sigma = self.sigma_i

    def forward(self, context_images, context_poses, poses):
        return self.gqn.generate(context_images, context_poses, poses)

    def training_step(self, batch, batch_idx):
        x_q, v_q = batch['query_image'], batch['query_pose']
        x, v = batch['context_images'], batch['context_poses']
        t = self.global_step
        sigma = max(self.sigma_f + (self.sigma_i - self.sigma_f)*(1 - t/(2e5)), self.sigma_f)
        elbo = self.gqn(x, v, v_q, x_q, sigma)
        loss = -elbo.mean()
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_test, v_test, x_q_test, v_q_test = batch['context_images'], batch['context_poses'], batch['query_image'], batch['query_pose']
        t = self.global_step
        sigma = max(self.sigma_f + (self.sigma_i - self.sigma_f)*(1 - t/(2e5)), self.sigma_f)
        elbo_test = self.gqn(x_test, v_test, v_q_test, x_q_test, sigma)
        kl_test = self.gqn.kl_divergence(x_test, v_test, v_q_test, x_q_test)
        x_q_rec_test = self.gqn.reconstruct(x_test, v_test, v_q_test, x_q_test)
        x_q_hat_test = self.gqn.generate(x_test, v_test, v_q_test)
        loss = -elbo_test.mean()
        self.log('test_kl', kl_test.mean())
        self.log('test_loss', loss)
        return dict(loss=loss, generated_image=x_q_hat_test, reconstructed_image=x_q_rec_test)

    def configure_optimizers(self):
        # NOTE: We will use 10 times more computational power
        # lr has to be updated accordingly
        # Original lr was 5e-4 -> 5e-5 (1.6e6 steps)
        optimizer = torch.optim.Adam(self.parameters(), lr=15e-4, betas=(0.9, 0.999), eps=1e-8)
        scheduler = AnnealingStepLR(optimizer, mu_i=15e-4, mu_f=15e-5, n=1.6e5)
        return [optimizer], [dict(scheduler=scheduler, interval='step', name='lr')]


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=args.batch_size)
    val_loader = DataLoader(mnist_val, batch_size=args.batch_size)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    model = LitAutoEncoder()

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == '__main__':
    cli_main()
