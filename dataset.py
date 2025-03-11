import os
from PIL import Image
from torchvision.datasets import VisionDataset

class xbdDataset(VisionDataset):
    def __init__(self, root = None, transforms = None, transform = None, target_transform = None):
        super(xbdDataset, self).__init__(root, transforms, transform, target_transform)
        self.images_dir = os.path.join(root, "images")
        self.masks_dir = os.path.join(root, "targets")
        self.image_filenames = sorted(os.listdir(self.images_dir))
        self.mask_filenames = sorted(os.listdir(self.masks_dir))
        assert len(self.image_filenames) == len(self.mask_filenames)

    def __len__(self):
        return len(self.mask_filenames)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.images_dir, self.image_filenames[index])
        mask_path = os.path.join(self.masks_dir, self.mask_filenames[index])

        image = Image.open(image_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        else:
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                mask = self.target_transform(mask)

        return image, mask

