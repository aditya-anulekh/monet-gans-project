import os.path as osp
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm
from dataset import PhotoMonetDataset
from discriminator_model import Discriminator
from generator_model import Generator
import config


def train_gan(disc_p, disc_m, gen_p, gen_m, dataloader, opt_disc, opt_gen, l1, mse):
    for idx, (monet_img, photo_img) in enumerate(dataloader):
        monet_img = monet_img.to(config.DEVICE)
        photo_img = photo_img.to(config.DEVICE)

        # Discriminator for Monet styled images from photos
        fake_monet = gen_m(photo_img)
        disc_monet_fake = disc_m(fake_monet.detach())
        disc_monet_real = disc_m(monet_img)
        disc_m_fake_loss = mse(disc_monet_fake, torch.zeros_like(disc_monet_fake))
        disc_m_real_loss = mse(disc_monet_real, torch.ones_like(disc_monet_real))
        disc_m_loss = disc_m_fake_loss + disc_m_real_loss

        # Discriminator for Photos from Monet styled paintings
        fake_photo = gen_p(monet_img)
        disc_photo_fake = disc_p(fake_photo.detach())
        disc_photo_real = disc_p(photo_img)
        disc_p_fake_loss = mse(disc_photo_fake, torch.zeros_like(disc_photo_fake))
        disc_p_real_loss = mse(disc_photo_real, torch.ones_like(disc_photo_real))
        disc_p_loss = disc_p_fake_loss + disc_p_real_loss

        disc_loss = (disc_m_loss + disc_p_loss)/2

        # Train discriminators
        opt_disc.zero_grad()
        disc_loss.backward()
        opt_disc.step()

        # Generators
        disc_m_fake = disc_m(monet_img)
        disc_p_fake = disc_p(photo_img)
        gen_m_loss = mse(disc_m_fake, torch.ones_like(disc_m_fake))
        gen_p_loss = mse(disc_p_fake, torch.ones_like(disc_p_fake))

        # Cycle loss - Fake monet should also generate a photo and vice versa
        cycle_monet = gen_m(fake_photo)
        cycle_photo = gen_p(fake_monet)
        cycle_monet_loss = l1(monet_img, cycle_monet)
        cycle_photo_loss = l1(photo_img, cycle_photo)

        # Identity loss - Monet fed to a Monet generator should result in a monet and likewise for photo
        identity_monet = gen_m(monet_img)
        identity_photo = gen_p(photo_img)
        identity_monet_loss = l1(monet_img, identity_monet)
        identity_photo_loss = l1(photo_img, identity_photo)

        total_gen_loss = gen_m_loss + gen_p_loss + cycle_monet_loss + cycle_photo_loss + identity_monet_loss + identity_photo_loss

        # Train generators
        opt_gen.zero_grad()
        total_gen_loss.backward()
        opt_gen.step()

        if idx%10 == 0:
            save_image(fake_monet, f"./generated_images/monet_{idx}.png")
            save_image(fake_photo, f"./generated_images/photo_{idx}.png")



if __name__ == "__main__":
    disc_p = Discriminator().to(config.DEVICE)
    disc_m = Discriminator().to(config.DEVICE)

    gen_p = Generator(in_channels=3).to(config.DEVICE)
    gen_m = Generator(in_channels=3).to(config.DEVICE)

    opt_disc = torch.optim.Adam(list(disc_m.parameters())+list(disc_p.parameters()),
                                lr=config.LEARNING_RATE,
                                betas=(0.5, 0.999))

    opt_gen = torch.optim.Adam(list(gen_m.parameters())+list(gen_p.parameters()),
                                lr=config.LEARNING_RATE,
                                betas=(0.5, 0.999))

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    dataset = PhotoMonetDataset(osp.join(config.DATASET_ROOT, "monet_jpg"), 
                                osp.join(config.DATASET_ROOT, "photo_jpg"))

    dataloader = DataLoader(dataset, 64, shuffle=True, num_workers=4)

    for i in range(config.NUM_EPOCHS):
        train_gan(disc_p, disc_m, gen_p, gen_m, dataloader, opt_disc, opt_gen, L1, mse)