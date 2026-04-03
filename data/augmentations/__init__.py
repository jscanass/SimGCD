from torchvision import transforms

import torch

def get_transform(transform_type='imagenet', image_size=32, args=None):

    if transform_type == 'imagenet':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation = args.interpolation
        crop_pct = args.crop_pct

        train_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
    
    else:

        raise NotImplementedError

    return (train_transform, test_transform)


def get_spectrogram_transform(
    image_size: int = 224,
    *,
    use_random_erasing: bool = True,
    random_erasing_p: float = 0.25,
    random_erasing_scale=(0.02, 0.08),
    random_erasing_ratio=(0.2, 3.3),
):
    """
    Spectrogram-safe transforms for bioacoustics time–frequency images.

    - No flips (would reverse time) and no ColorJitter (would distort magnitudes).
    - Uses ImageNet normalization to match pretrained ViT expectations.
    - Optional mild RandomErasing approximates SpecAugment-style masking.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_ops = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    ]
    if use_random_erasing and random_erasing_p > 0:
        train_ops.append(
            transforms.RandomErasing(
                p=float(random_erasing_p),
                scale=random_erasing_scale,
                ratio=random_erasing_ratio,
                value=0,
            )
        )

    train_transform = transforms.Compose(train_ops)
    test_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]
    )

    return train_transform, test_transform

