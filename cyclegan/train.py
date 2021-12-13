from datetime import datetime
import os
import os.path as osp
import warnings
import logging

import torch
import torch.nn as nn
from torch.serialization import save
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from dataset import PhotoMonetDataset
from discriminator_model import Discriminator
from generator_model import Generator
import config
import pickle as pkl


def train_gan(disc_p, disc_m, gen_p, gen_m, train_dataloader, opt_disc, opt_gen, l1, mse):
    for idx, (monet_img, photo_img) in enumerate(tqdm(train_dataloader)):
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
        disc_m_fake = disc_m(fake_monet)
        disc_p_fake = disc_p(fake_photo)
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

        total_gen_loss = (gen_m_loss + gen_p_loss 
                        + config.LAMBDA*(cycle_monet_loss + cycle_photo_loss) 
                        + config.LAMBDA_IDENTITY*(identity_monet_loss + identity_photo_loss))

        # Train generators
        opt_gen.zero_grad()
        total_gen_loss.backward()
        opt_gen.step()

        if idx%500 == 0:
            with open(f"{CHECKPOINT_DIR}/logs.txt", "a") as file:
                file.write(f"loss - gen={total_gen_loss:0.2f}, disc={disc_loss:0.2f}\n")
    
    # Mark the end of an epoch
    with open(f"{CHECKPOINT_DIR}/logs.txt", "a") as file:
        file.write(f"*"*50)


def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)


if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Check if running in DEBUG mode
    if config.DEBUG:
        warnings.warn("Running in Debug model. Nothing will be saved. Set DEBUG to True in cyclegan/config.py")
    disc_p = Discriminator().to(config.DEVICE)
    disc_m = Discriminator().to(config.DEVICE)

    gen_p = Generator(in_channels=3).to(config.DEVICE)
    gen_m = Generator(in_channels=3).to(config.DEVICE)

    disc_p.apply(initialize_weights)
    disc_m.apply(initialize_weights)

    gen_p.apply(initialize_weights)
    gen_m.apply(initialize_weights)

    if config.LOAD_CHECKPOINT:
        disc_p.load_state_dict(torch.load("cyclegan/checkpoints/1212210846/disc_p_31.pth"))
        disc_m.load_state_dict(torch.load("cyclegan/checkpoints/1212210846/disc_m_31.pth"))

        gen_p.load_state_dict(torch.load("cyclegan/checkpoints/1212210846/gen_p_31.pth"))
        gen_m.load_state_dict(torch.load("cyclegan/checkpoints/1212210846/gen_m_31.pth"))

        print("Loaded all checkpoints successfully")

    opt_disc = torch.optim.Adam(list(disc_m.parameters())+list(disc_p.parameters()),
                                lr=config.LEARNING_RATE,
                                betas=(0.5, 0.999))

    opt_gen = torch.optim.Adam(list(gen_m.parameters())+list(gen_p.parameters()),
                                lr=config.LEARNING_RATE,
                                betas=(0.5, 0.999))

    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5 for _ in range(3)],
                                    [0.5 for _ in range(3)]),
            ])
    train_dataset = PhotoMonetDataset(osp.join(config.DATASET_ROOT, "trainB"), 
                                osp.join(config.DATASET_ROOT, "trainA"),
                                transform=transform)

    test_dataset = PhotoMonetDataset(osp.join(config.DATASET_ROOT, "testB"), 
                                osp.join(config.DATASET_ROOT, "testA"),
                                transform=transform)

    print(train_dataset.__len__())
    print(test_dataset.__len__())
    train_dataloader = DataLoader(train_dataset, 1, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4)

    # writer_inputs = SummaryWriter("cyclegan/logs/inputs")
    # writer_monet = SummaryWriter("cyclegan/logs/monet")
    # writer_photo = SummaryWriter("cyclegan/logs/photo")

    # Same test images to monitor the performance of the GAN
    monet_img, photo_img = next(iter(test_dataloader))
    monet_img = monet_img.to(config.DEVICE)
    photo_img = photo_img.to(config.DEVICE)
    
    save_image(monet_img, "cyclegan/generated_images/monet_og.png")
    save_image(photo_img, "cyclegan/generated_images/photo_og.png")

    # Create directory to store images
    CHECKPOINT_DIR = f"cyclegan/checkpoints/{datetime.now().strftime('%d%m%y%H%M')}"
    if not osp.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)
        print(f"Created checkpoint dir at {CHECKPOINT_DIR}")

    # Write existing configs to logs.txt file
    with open(f"{CHECKPOINT_DIR}/logs.txt", "w") as file:
        file.write(f"Cycle consistency loss = {config.LAMBDA}\n")
        file.write(f"Identity loss = {config.LAMBDA_IDENTITY}\n")

    for i in range(config.NUM_EPOCHS):
        print(f"Epoch - {i}/{config.NUM_EPOCHS}")
        train_gan(disc_p, disc_m, gen_p, gen_m, train_dataloader, opt_disc, opt_gen, L1, mse)
        
        # Save all model weights

        torch.save(disc_m.state_dict(), osp.join(CHECKPOINT_DIR, f'disc_m_{i}.pth'))
        torch.save(disc_p.state_dict(), osp.join(CHECKPOINT_DIR, f'disc_p_{i}.pth'))
        torch.save(gen_m.state_dict(), osp.join(CHECKPOINT_DIR, f'gen_m_{i}.pth'))
        torch.save(gen_p.state_dict(), osp.join(CHECKPOINT_DIR, f'gen_p_{i}.pth'))

        if (i<20 or i%10==0):
            with torch.no_grad():
                fake_monet = gen_m(photo_img)
                fake_photo = gen_p(monet_img)
                with open("generated.pkl", "wb") as file:
                    pkl.dump([fake_monet, fake_photo], file)
                save_image(fake_monet*0.5+0.5, f"cyclegan/generated_images/monet_{i}.png")
                save_image(fake_photo*0.5+0.5, f"cyclegan/generated_images/photo_{i}.png")

                # input_photos = make_grid(torch.cat((monet_img, photo_img)), normalize=True)
                # monet_generated = make_grid(torch.cat((photo_img, fake_monet*0.5+0.5)), normalize=False)
                # photo_generated = make_grid(torch.cat((monet_img, fake_photo*0.5+0.5)), normalize=False)
                # writer_inputs.add_image("Inputs", input_photos, global_step=i*75)
                # writer_monet.add_images("Monet", 
                #                         torch.cat((photo_img, fake_monet*0.5+0.5)), 
                #                         global_step=i*75)
                # writer_photo.add_images("Photo", 
                #                         torch.cat((monet_img, fake_photo*0.5+0.5)), 
                #                         global_step=i*75)
