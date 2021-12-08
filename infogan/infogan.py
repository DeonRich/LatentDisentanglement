import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
from torch.autograd import Variable
from module import *

import torch.nn as nn
import torch.nn.functional as F
import torch
def parse():
    os.makedirs("images/static/", exist_ok=True)
    os.makedirs("weights/", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=600, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=32, help="dimensionality of the latent space")
    parser.add_argument("--code_dim", type=int, default=32, help="latent code")
    parser.add_argument("--n_classes", type=int, default=120, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=1200, help="interval between image sampling")
    parser.add_argument("--load", type=int, default=None, help="interval between image sampling")
    opt = parser.parse_args()
    #for i in range(1, opt.code_dim+1):
    os.makedirs(f"images/varying_c/", exist_ok=True)
    print(opt)
    return opt
cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(FloatTensor(y_cat))

class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, code_dim, img_size):
        super(Generator, self).__init__()
        input_dim = latent_dim + n_classes + code_dim

        ngf = 128
        self.init_size = img_size // 16  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(input_dim, ngf * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            GBlock(ngf,      ngf >> 1, upsample=True),
            GBlock(ngf >> 1, ngf >> 2, upsample=True),
            GBlock(ngf >> 2, ngf >> 3, upsample=True),
            GBlock(ngf >> 3, ngf >> 4, upsample=True),
            nn.BatchNorm2d(ngf >> 4),
            nn.Conv2d(ngf >> 4, 3, 3, 1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size, n_classes, code_dim):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        ndf = 1024
        self.conv_blocks = nn.Sequential(
            DBlockOptimized(3, ndf >> 4),
            DBlock(ndf >> 4, ndf >> 3, downsample=True),
            DBlock(ndf >> 3, ndf >> 2, downsample=True),
            DBlock(ndf >> 2, ndf >> 1, downsample=True),
            DBlock(ndf >> 1, ndf, downsample=True),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(ndf, 1))
        self.aux_layer = nn.Sequential(nn.Linear(ndf, n_classes), nn.Softmax())
        self.latent_layer = nn.Sequential(nn.Linear(ndf, code_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = torch.sum(out, dim=(2,3))
        #out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code

def train(opt):
    # Loss functions
    adversarial_loss = torch.nn.MSELoss()
    categorical_loss = torch.nn.CrossEntropyLoss()
    continuous_loss = torch.nn.MSELoss()

    # Loss weights
    lambda_cat = 1
    lambda_con = 0.1

    # Initialize generator and discriminator

    generator = Generator(opt.latent_dim, opt.n_classes, opt.code_dim, opt.img_size)
    discriminator = Discriminator(opt.img_size, opt.n_classes, opt.code_dim)
    offset = 0 if opt.load is None else opt.load + 1 
    if opt.load is not None:
        generator.load_state_dict(torch.load(f"weights/generator_{opt.load:07}.weights"))
        discriminator.load_state_dict(torch.load(f"weights/discriminator_{opt.load:07}.weights"))
        print("Loaded")

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        categorical_loss.cuda()
        continuous_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Configure data loader
    os.makedirs("../../data/mnist", exist_ok=True)
    # dataloader = torch.utils.data.DataLoader(
    #     datasets.MNIST(
    #         "../../data/mnist",
    #         train=True,
    #         download=True,
    #         transform=transforms.Compose(
    #             [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    #         ),
    #     ),
    #     batch_size=opt.batch_size,
    #     shuffle=True,
    # )
    dataImages = np.load("pre_imgs.npy")
    dataLabels = np.load("pre_labels.npy")
    dataloader = torch.utils.data.DataLoader(
        TensorDataset(
            torch.tensor(dataImages),
            torch.tensor(dataLabels).long()
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )


    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_info = torch.optim.Adam(
        itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    # Static generator inputs for sampling
    #num_static_samples = opt.n_classes
    num_static_samples = 10
    static_choices = np.random.choice(opt.n_classes, num_static_samples, replace=False)
    static_z = Variable(FloatTensor(np.zeros((num_static_samples ** 2, opt.latent_dim))))
    static_label = to_categorical(
        np.array([num for _ in range(num_static_samples) for num in static_choices]), num_columns=opt.n_classes
    )
    static_code = Variable(FloatTensor(np.zeros((num_static_samples ** 2, opt.code_dim))))

    def sample_image(n_row, batches_done):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Static sample
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
        static_sample = generator(z, static_label, static_code)
        save_image(static_sample.data, "images/static/%d.png" % batches_done, nrow=n_row, normalize=True)

        # Get varied c1 and c2
        zeros = np.zeros((n_row ** 2, opt.code_dim-1))
        c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
        c = Variable(FloatTensor(np.concatenate((c_varied, zeros), -1)))
        for i in np.random.choice(opt.code_dim, 2):
            sample = generator(static_z, static_label, torch.roll(c, i, dims=-1))
            save_image(sample.data, f"images/varying_c/{batches_done:07}_c{i+1:03}.png", nrow=n_row, normalize=True)


    # ----------
    #  Training
    # ----------

    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):

            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = to_categorical(labels.numpy(), num_columns=opt.n_classes)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            label_input = to_categorical(np.random.randint(0, opt.n_classes, batch_size), num_columns=opt.n_classes)
            code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

            # Generate a batch of images
            gen_imgs = generator(z, label_input, code_input)

            # Loss measures generator's ability to fool the discriminator
            validity, _, _ = discriminator(gen_imgs)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            real_pred, _, _ = discriminator(real_imgs)
            d_real_loss = adversarial_loss(real_pred, valid)

            # Loss for fake images
            fake_pred, _, _ = discriminator(gen_imgs.detach())
            d_fake_loss = adversarial_loss(fake_pred, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # ------------------
            # Information Loss
            # ------------------

            optimizer_info.zero_grad()

            # Sample labels
            sampled_labels = np.random.randint(0, opt.n_classes, batch_size)

            # Ground truth labels
            gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)

            # Sample noise, labels and code as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)
            code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

            gen_imgs = generator(z, label_input, code_input)
            _, pred_label, pred_code = discriminator(gen_imgs)

            info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
                pred_code, code_input
            )

            info_loss.backward()
            optimizer_info.step()

            # --------------
            # Log Progress
            # --------------

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), info_loss.item())
            )
            batches_done = epoch * len(dataloader) + i + offset
            if batches_done % opt.sample_interval == 0:
                torch.save(generator.state_dict(), f"weights/generator_{batches_done:07}.weights")
                torch.save(discriminator.state_dict(), f"weights/discriminator_{batches_done:07}.weights")
                sample_image(n_row=num_static_samples, batches_done=batches_done)

if __name__ == '__main__':
    train(parse())