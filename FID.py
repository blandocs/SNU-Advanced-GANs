# Official FID code: https://github.com/bioinf-jku/TTUR

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

try:
    import nsml
    from nsml import DATASET_PATH, SESSION_NAME
except ImportError:
    nsml = None

def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)

@torch.no_grad()
def extract_feature_from_generated_samples(
    inception, batch_size, info
):

    n_sample, device = info['n_sample'], info['device']

    transform = transforms.Compose([
        transforms.Resize([info['image_height'], info['image_width']]),
        transforms.Resize([info['height'], info['width']]),
        transforms.ToTensor(),
        transforms.Normalize(mean=info['mean'], std=info['std'])
    ])

    features = []

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
    loader = data.DataLoader(dataset=cifar10, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # generated images should match with n sample
    print(len(loader), n_sample, batch_size)

    if n_sample % batch_size == 0:
        assert len(loader) == n_sample // batch_size
    else:
        assert len(loader) == n_sample // batch_size + 1

    for i, (images, _) in enumerate(tqdm(loader)):
        images = images.to(device)
        feat = inception(images)[0].view(images.shape[0], -1)
        features.append(feat.to('cpu'))

    features = torch.cat(features, 0)
    print("The number of feature: ", len(features))
    assert len(features) == n_sample

    return features

@torch.no_grad()
def extract_feature_from_samples(
    model, inception, batch_size, info
):
    
    n_sample, device, latent_size, image_height, image_width, image_channel, height, width, mean, std, network = info['n_sample'], info['device'], info['latent_size'], info['image_height'], info['image_width'], info['image_channel'], info['height'], info['width'], info['mean'], info['std'], info['network']


    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    if resid > 0:
        batch_sizes = [batch_size] * n_batch + [resid]
    else:
        batch_sizes = [batch_size] * n_batch
        
    features = []

    for batch in tqdm(batch_sizes, mininterval=1):
        if network == 'GAN':
            z = torch.randn(batch, latent_size).to(device)
        elif network == 'DCGAN':
            z = torch.randn(batch, latent_size, 1, 1, device=device)

        imgs = model(z)
        imgs = (imgs + 1) / 2 # -1 ~ 1 to 0~1
        imgs = torch.clamp(imgs, 0, 1, out=None)

        imgs = imgs.reshape(imgs.size(0), image_channel, image_height, image_width)
        imgs = F.interpolate(imgs, size=(height, width), mode='bilinear', align_corners=True)

        transformed = []
        for img in imgs:
            transformed.append(transforms.Normalize(mean=mean, std=std)(img))
        transformed = torch.stack(transformed, dim=0)
        # assert transformed.shape == imgs.shape

        feat = inception(transformed)[0].view(imgs.shape[0], -1)
        features.append(feat.to('cpu'))

    features = torch.cat(features, 0)

    return features


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)
    
    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


@torch.no_grad()
def get_fid(model, batch, info):
    n_sample, device, real_mean_cov = info['n_sample'], info['device'], info['real_mean_cov']
    inception = load_patched_inception_v3().to(device)
    if device == 'cuda':
        inception = nn.DataParallel(inception)
    inception.eval()

    info['height'], info['width'], info['mean'], info['std'] = 299, 299, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if not os.path.exists(real_mean_cov):
        # calculate mean and cov                
        features = extract_feature_from_generated_samples(
            inception, batch, info
        ).numpy()

        sample_mean = np.mean(features, 0)
        sample_cov = np.cov(features, rowvar=False)

        with open(real_mean_cov, 'wb') as handle:
            pickle.dump({'mean': sample_mean, 'cov': sample_cov}, handle)

    with open(real_mean_cov, 'rb') as f:
        embeds = pickle.load(f)
        real_mean = embeds['mean']
        real_cov = embeds['cov']
    
    if model:
        model.eval()
        features = extract_feature_from_samples(
            model, inception, batch, info
        ).numpy()

        print(f'extracted {features.shape[0]} features')
        sample_mean = np.mean(features, 0)
        sample_cov = np.cov(features, rowvar=False)

        fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
        print('fid:', fid)
        model.train()
        return fid
    else:
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--n_sample', type=int, default=50000)
    parser.add_argument('--network', type=str, choices=['GAN', 'DCGAN'])
    args = parser.parse_args()

    # Hyper-parameters
    if args.network == 'GAN':   
        image_height, image_width, image_channel = 32, 32, 3
    elif args.network == 'DCGAN':   
        image_height, image_width, image_channel = 64, 64, 3
    else:
        print('Select the target network')
        exit()

    info = {
        'n_sample': args.n_sample,
        'device': args.device,
        'image_height': image_height,
        'image_width': image_width,
        'image_channel': image_channel,
        'real_mean_cov': f'real_mean_cov_{image_height}.pkl'
    }

    fid = get_fid(None, args.batch, info)
