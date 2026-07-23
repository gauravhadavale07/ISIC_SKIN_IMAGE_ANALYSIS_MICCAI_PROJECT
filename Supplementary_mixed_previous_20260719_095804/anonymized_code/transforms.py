from torchvision import transforms
from config import cfg

def get_train_transforms() -> transforms.Compose:
    """
    Robust data augmentation policy for dermatology.
    Prevents visual memorization and forces learning of morphological invariants.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            cfg.data.img_size,
            scale=(0.8, 1.0),   # Lesion must stay in frame
            ratio=(0.9, 1.1)    # Near-square crops only for dermoscopy
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        # Simulates varied clinical lighting and dermatoscope intensity
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.img_mean, std=cfg.data.img_std)
    ])

def get_eval_transforms() -> transforms.Compose:
    """
    Strict, deterministic spatial transforms for Validation and zero-shot OOD Testing.
    """
    return transforms.Compose([
        transforms.Resize((cfg.data.img_size, cfg.data.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.img_mean, std=cfg.data.img_std)
    ])