import os
import time
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, resize
from PIL import Image
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True

# Paths
data_path = "/home/shared/data_raw/LAC/data_collection_1"
img_dir = os.path.join(data_path, 'front_left')
mask_dir = os.path.join(data_path, 'front_left_semantic')

# Semantic mapping (RGB)
semantic_colors = {
    (250, 170, 30): 0,
    (108, 59, 42): 1,
    (110, 190, 160): 2,
    (81, 0, 81): 3,
    (0, 0, 0): 4,
}

color_map = np.zeros((256, 256, 3), dtype=np.uint8)
color_to_index = {tuple(k): v for k, v in semantic_colors.items()}

def rgb_to_class(mask_rgb):
    mask_np = np.array(mask_rgb)
    class_mask = np.zeros(mask_np.shape[:2], dtype=np.uint8)
    for rgb, idx in color_to_index.items():
        class_mask[np.all(mask_np == rgb, axis=-1)] = idx
    return class_mask

class SegDataset(Dataset):
    def __init__(self, filenames):
        self.img_paths = [os.path.join(img_dir, f) for f in filenames]
        self.mask_paths = [os.path.join(mask_dir, f) for f in filenames]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        mask = cv2.imread(self.mask_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        img_tensor = to_tensor(resize(Image.fromarray(img), (256, 256)))
        mask_class = rgb_to_class(mask)
        mask_tensor = torch.from_numpy(np.array(resize(Image.fromarray(mask_class), (256, 256), Image.NEAREST))).long()
        return img_tensor, mask_tensor

# File split
all_filenames = [f"{i}.png" for i in range(34, 1794)]
np.random.seed(42)
np.random.shuffle(all_filenames)
split_idx = int(0.8 * len(all_filenames))
train_files = all_filenames[:split_idx]
test_files = all_filenames[split_idx:]

train_ds = SegDataset(train_files)
test_ds = SegDataset(test_files)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=24, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=4, num_workers=24, pin_memory=True)

model_names = [
    'Unet', 'UnetPlusPlus', 'FPN', 'PSPNet',
    'DeepLabV3', 'DeepLabV3Plus', 'Linknet',
    'MAnet', 'PAN'
]

device = torch.device('cuda')
results = {}

for name in model_names:
    print(f"\n==> Training {name}")
    model = getattr(smp, name)(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=5,
    ).to(device).to(memory_format=torch.channels_last)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    for epoch in range(5):
        total_loss = 0
        for images, masks in tqdm(train_loader):
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    eval_loss = 0
    start_time = time.time()
    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            masks = masks.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, masks)
            eval_loss += loss.item()
    total_time = time.time() - start_time
    secs_per_image = total_time / len(test_loader.dataset)

    # Visualization (only first batch)
    with torch.no_grad():
        for images, masks in  test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu()
            images = images.cpu().permute(0, 2, 3, 1).numpy()
            masks = masks.cpu().numpy()

            fig, axs = plt.subplots(len(images), 3, figsize=(12, 4 * len(images)))
            for i in range(len(images)):
                axs[i, 0].imshow(images[i])
                axs[i, 0].set_title("Image")
                axs[i, 1].imshow(masks[i])
                axs[i, 1].set_title("Mask")
                axs[i, 2].imshow(preds[i])
                axs[i, 2].set_title("Prediction")
                for ax in axs[i]:
                    ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"./segmentation_output/{name}_segmentation_batch.png")
            plt.close()
            break

    results[name] = {
        'test_loss': eval_loss / len(test_loader),
        'secs_per_image': secs_per_image
    }

print("\n=== Results ===")
for model_name, metrics in results.items():
    print(f"{model_name:15s} | Loss: {metrics['test_loss']:.4f} | Secs/img: {metrics['secs_per_image']:.4f}")