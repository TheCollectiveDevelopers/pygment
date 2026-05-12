import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from image import rgb_to_cmyk, soft_proof, apply_hsl_offsets_torch
import os
import time

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" #type: ignore


class ImageDataset(Dataset):
    def __init__(self, image_dir: str, return_og: bool = False):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg' ,'.JPG', '.png'))]
        self.transform = T.Compose([T.ToTensor()])
        self.return_og = return_og

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        
        if self.return_og:
            og_image = image.copy()
            og_cmyk_image = rgb_to_cmyk(og_image)
            og_soft_proofed_image = soft_proof(og_cmyk_image)
            og_tensor = self.transform(og_image)
            og_soft_tensor = self.transform(og_soft_proofed_image)
        else:
            og_tensor = torch.empty(0)
            og_soft_tensor = torch.empty(0)
            
        image = image.resize((50, 50), resample=Image.BILINEAR)

        cmyk_image = rgb_to_cmyk(image)
        soft_proofed_image = soft_proof(cmyk_image)

        rgb_tensor = self.transform(image)
        cmyk_tensor = self.transform(cmyk_image)
        soft_tensor = self.transform(soft_proofed_image)

        return rgb_tensor, cmyk_tensor, soft_tensor, og_tensor, og_soft_tensor


def collate_batch(batch):
    rgb, cmyk, soft, og_rgb, og_soft = zip(*batch)
    return (
        torch.stack(rgb, dim=0),
        torch.stack(cmyk, dim=0),
        torch.stack(soft, dim=0),
        list(og_rgb),
        list(og_soft),
    )

training_data = DataLoader(
    ImageDataset("./dataset/", return_og=False),
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_batch,
)

test_data = DataLoader(
    ImageDataset("./testd/", return_og=True),
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_batch,
)

class HSLPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 24)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.mlp(features)


def scale_hsl_offsets(raw: torch.Tensor) -> torch.Tensor:
    raw = raw.view(-1, 8, 3)
    raw = torch.tanh(raw)

    hue_shift_deg = 12.0
    l_offset_max = 0.2
    s_offset_max = 0.2

    h_shift = raw[..., 0] * (hue_shift_deg / 360.0)
    l_offset = raw[..., 1] * l_offset_max
    s_offset = raw[..., 2] * s_offset_max
    return torch.stack([h_shift, l_offset, s_offset], dim=-1)


def visualize_samples(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    output_dir: str,
    epoch: int,
    max_items: int = 4,
) -> None:
    model.eval()
    batch = next(iter(loader), None)
    if batch is None:
        return

    _rgb, cmyk, _soft, og_rgb, og_soft = batch
    cmyk = cmyk.to(device)

    with torch.no_grad():
        raw_offsets = model(cmyk)
        offsets = scale_hsl_offsets(raw_offsets)

    to_pil = T.ToPILImage()
    os.makedirs(output_dir, exist_ok=True)

    count = min(max_items, len(og_rgb))
    for i in range(count):
        og_rgb_tensor = og_rgb[i]
        og_soft_tensor = og_soft[i]

        pred = apply_hsl_offsets_torch(
            og_soft_tensor.unsqueeze(0).to(device),
            offsets[i : i + 1],
        ).squeeze(0).cpu()

        img_rgb = to_pil(og_rgb_tensor)
        img_pred = to_pil(pred)
        img_soft = to_pil(og_soft_tensor)

        width = img_rgb.width + img_pred.width + img_soft.width
        height = max(img_rgb.height, img_pred.height, img_soft.height)
        canvas = Image.new("RGB", (width, height))
        canvas.paste(img_rgb, (0, 0))
        canvas.paste(img_soft, (img_rgb.width, 0))
        canvas.paste(img_pred, (img_rgb.width + img_soft.width, 0))

        out_path = os.path.join(output_dir, f"epoch_{epoch:03d}_sample_{i}.png")
        canvas.save(out_path)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader | None,
    epochs: int,
    device: str,
    output_dir: str,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        print(f"Starting epoch {epoch}/{epochs}...")
        epoch_start_time = time.time()
        
        train_start_time = time.time()
        model.train()
        running_loss = 0.0

        for rgb, cmyk, soft, _og_rgb, _og_soft in train_loader:
            rgb = rgb.to(device)
            cmyk = cmyk.to(device)
            soft = soft.to(device)

            optimizer.zero_grad(set_to_none=True)
            raw_offsets = model(cmyk)
            offsets = scale_hsl_offsets(raw_offsets)
            pred = apply_hsl_offsets_torch(soft, offsets)
            loss = criterion(pred, rgb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * rgb.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        train_time = time.time() - train_start_time
        print(f"Epoch {epoch}/{epochs} - train loss: {avg_loss:.6f} (Time: {train_time:.2f}s)")

        if test_loader is not None:
            test_start_time = time.time()
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for rgb, cmyk, soft, _og_rgb, _og_soft in test_loader:
                    rgb = rgb.to(device)
                    cmyk = cmyk.to(device)
                    soft = soft.to(device)

                    raw_offsets = model(cmyk)
                    offsets = scale_hsl_offsets(raw_offsets)
                    pred = apply_hsl_offsets_torch(soft, offsets)
                    loss = criterion(pred, rgb)
                    test_loss += loss.item() * rgb.size(0)

            avg_test_loss = test_loss / len(test_loader.dataset)
            test_time = time.time() - test_start_time
            print(f"Epoch {epoch}/{epochs} - test loss: {avg_test_loss:.6f} (Time: {test_time:.2f}s)")
            
            vis_start_time = time.time()
            visualize_samples(model, test_loader, device, output_dir, epoch)
            vis_time = time.time() - vis_start_time
            print(f"Epoch {epoch}/{epochs} - visualization time: {vis_time:.2f}s")
            
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch}/{epochs} - total time: {epoch_time:.2f}s")


if __name__ == "__main__":
    model = HSLPredictor().to(device)
    train(
        model=model,
        train_loader=training_data,
        test_loader=test_data,
        epochs=10,
        device=device,
        output_dir="./outputs",
    )
