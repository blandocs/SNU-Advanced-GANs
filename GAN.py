# Refer to https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/generative_adversarial_network/main.py
# python GAN.py
# Command line: nsml run -d cifar10 -e "GAN.py" -g 1 -c 10 --memory "20G"
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_20-03.html#rel_20-03 -> pytorch 1.5.0

import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from calc_inception import load_patched_inception_v3
from FID import get_fid

try:
    import nsml
    from nsml import DATASET_PATH, SESSION_NAME
except ImportError:
    nsml = None

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
latent_size = 64
hidden_size = 256
image_height, image_width, image_channel = 32, 32, 3
image_size = image_height*image_width*image_channel
num_epochs = 3000
batch_size = 100
sample_dir = 'samples'
ckpt_dir = 'checkpoints'

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# Image processing
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
                                     std=(0.5, 0.5, 0.5))])
                  
# CIFAR10 dataset
if nsml:
    data_dir = os.path.join(DATASET_PATH, 'train')
else:
    data_dir = 'data/'

cifar10 = torchvision.datasets.CIFAR10(root=data_dir,
                                   train=True,
                                   transform=transform,
                                   download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=cifar10,
                                          batch_size=batch_size, 
                                          shuffle=True)

# Discriminator
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())

# Generator 
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())

# Device setting
D = D.to(device)
G = G.to(device)
inception = load_patched_inception_v3().eval().to(device)

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()

# Start training
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(batch_size, -1).to(device)
        
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        
        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels)
        
        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))
            exit()
    # Save real images
    if (epoch+1) == 1:
        images = images.reshape(images.size(0), image_channel, image_height, image_width)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))

    if (epoch+1) % 100 == 1:
        info = {
            'network': 'GAN',
            'n_sample': 50000,
            'device': device,
            'latent_size': latent_size,
            'image_height': image_height,
            'image_width': image_width,
            'image_channel': image_channel,
            'real_mean_cov': 'real_mean_cov_32_cifar10.pkl',
            'dataset': 'cifar10'
        }

        fid = get_fid(G, inception, batch_size, info)
        print(f'fid: {fid}')
            
        # save ckpt
        torch.save(G.state_dict(), f'{ckpt_dir}/G_epoch{epoch+1}.ckpt')
        torch.save(D.state_dict(), f'{ckpt_dir}/D_epoch{epoch+1}.ckpt')
        
        # nsml report
        if nsml:
            nsml.report(summary=True, step=epoch+1, fid=fid, d_loss=d_loss.item(), g_loss=g_loss.item(), 
            real_score=real_score.mean().item(), fake_score=fake_score.mean().item())
    
        # Save sampled images
        fake_images = fake_images.reshape(fake_images.size(0), image_channel, image_height, image_width)
        save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

# Save the model checkpoints 
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')