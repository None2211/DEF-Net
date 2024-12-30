from torchvision import transforms
import random
import torch

class RandomTransforms:
    def __init__(self):
        self.degrees = [90, 180, 270]  # Define the fixed degrees for rotation

    def apply_transforms(self, x, seed, degree):
        # Seed the random generator for consistent transformations
        random.seed(seed)
        torch.manual_seed(seed)

        # Define the transformations to apply
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Lambda(lambda x: x.rotate(degree)),
        ])

        # Apply transformations
        return transform(x)

    def __call__(self, image, superpixel_image, mask):
        # Generate a random state for this call
        seed = random.randint(0, 2**32 - 1)
        degree = random.choice(self.degrees)  # Choose a rotation degree

        # Apply the same transformations to image, superpixel_image, and mask
        image = self.apply_transforms(image, seed, degree)
        superpixel_image = self.apply_transforms(superpixel_image, seed, degree)
        mask = self.apply_transforms(mask, seed, degree)

        return image, superpixel_image, mask
