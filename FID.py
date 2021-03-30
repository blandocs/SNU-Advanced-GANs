# Official FID code: https://github.com/bioinf-jku/TTUR
# http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# python FID.py --network DCGAN --dataset celeba
# Refer to https://github.com/rosinality/stylegan2-pytorch/blob/master/fid.py
# Refer to https://github.com/clovaai/stargan-v2/blob/master/metrics/fid.py

import argparse
import pickle
import torch
from torch import nn
import numpy as np
from scipy import linalg
from calc_inception import load_patched_inception_v3
import torchvision
from torchvision import utils, transforms
from torch.nn import functional as F
from torch.utils import data
import os
from tqdm import tqdm
from dataset import DataSetFromDir

try:
    import nsml
    from nsml import DATASET_PATH, SESSION_NAME
except ImportError:
    nsml = None


@torch.no_grad()
def extract_feature_real(
    inception, batch_size, info
):
    n_sample, device, dataset_name = info['n_sample'], info['device'], info['dataset']

    transform = transforms.Compose([
        transforms.Resize([info['image_height'], info['image_width']]),
        transforms.Resize([info['imagenet_height'], info['imagenet_width']]),
        transforms.ToTensor(),
        transforms.Normalize(mean=info['mean'], std=info['std'])
    ])

    if dataset_name == 'cifar10':
        data_dir = os.path.join(DATASET_PATH, 'train') if nsml else 'data/'
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)
    elif dataset_name == 'celeba':
        data_dir = os.path.join(DATASET_PATH, 'train') if nsml else 'data/img_align_celeba'
        dataset = DataSetFromDir(data_dir, transform)
    
    dataset, _ = data.random_split(dataset, [n_sample, len(dataset)-n_sample])
    loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    features = []
    for i, (images, _) in enumerate(tqdm(loader)):
        images = images.to(device)
        feat = inception(images)[0].view(images.shape[0], -1)
        features.append(feat.to('cpu'))

    features = torch.cat(features, 0)
    print("The number of feature: ", len(features))
    assert len(features) == n_sample

    return features


@torch.no_grad()
def extract_feature_fake(
    model, inception, batch_size, info
):
    n_sample, device, latent_size, image_height, image_width, image_channel, height, width, mean, std, network = info['n_sample'], info['device'], info['latent_size'], info['image_height'], info['image_width'], info['image_channel'], info['imagenet_height'], info['imagenet_width'], info['mean'], info['std'], info['network']

    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid]  if resid > 0 else [batch_size] * n_batch

    features = []

    for batch in tqdm(batch_sizes, mininterval=1):
        z = torch.randn(batch, latent_size).to(device) if network == 'GAN' else torch.randn(batch, latent_size, 1, 1, device=device) # else for DCGAN
        imgs = model(z)
        imgs = (imgs + 1) / 2 # -1 ~ 1 to 0~1
        imgs = torch.clamp(imgs, 0, 1, out=None)
        imgs = imgs.reshape(imgs.size(0), image_channel, image_height, image_width)
        imgs = F.interpolate(imgs, size=(height, width), mode='bilinear', align_corners=True)

        transformed = []
        for img in imgs:
            transformed.append(transforms.Normalize(mean=mean, std=std)(img))
        transformed = torch.stack(transformed, dim=0)

        feat = inception(transformed)[0].view(imgs.shape[0], -1)
        features.append(feat.to('cpu'))

    features = torch.cat(features, 0)

    return features


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(np.dot(sample_cov, real_cov), disp=False)
    dist = np.sum((sample_mean -real_mean)**2) + np.trace(sample_cov + real_cov - 2*cov_sqrt)
    fid = np.real(dist) # Return the real part of the complex argument.
    return fid


@torch.no_grad()
def get_fid(model, inception, batch, info):
    n_sample, device, real_mean_cov = info['n_sample'], info['device'], info['real_mean_cov']
    info['imagenet_height'], info['imagenet_width'], info['mean'], info['std'] = 299, 299, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if not os.path.exists(real_mean_cov):
        print("Calculate mean and cov of the real dataset.")  
        features = extract_feature_real(inception, batch, info).numpy()
        sample_mean = np.mean(features, 0)
        sample_cov = np.cov(features, rowvar=False)

        with open(real_mean_cov, 'wb') as handle:
            pickle.dump({'mean': sample_mean, 'cov': sample_cov}, handle)
    else:
        print("Statistics of the real dataset are already exists.")  

    with open(real_mean_cov, 'rb') as f:
        embeds = pickle.load(f)
        real_mean = embeds['mean']
        real_cov = embeds['cov']
    
    model.eval()
    features = extract_feature_fake(model, inception, batch, info).numpy()
    model.train()

    print(f'extracted {features.shape[0]} features')
    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
    print('fid:', fid)
    
    return fid
