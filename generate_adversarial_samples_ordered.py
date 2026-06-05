"""
generate_adversarial_samples_ordered.py
----------------------------------------
Giống generate_adversarial_samples.py nhưng:
  1. Sort ảnh theo thứ tự numeric: 0, 1, 2, ..., 999
  2. Hỗ trợ batch_size > 1 (fix bug .squeeze() trong Base.py)

Cách chạy (1000 ảnh, batch 10):
    python generate_adversarial_samples_ordered.py \
        data.cle_data_path=resources/images/bigscale_1000 \
        data.tgt_data_path=resources/images/target_images_1000 \
        data.num_samples=1000 \
        data.batch_size=10
"""

import os
import re
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import hydra
from omegaconf import OmegaConf
from typing import List, Dict, Optional, Tuple
from torch import nn
from tqdm import tqdm
import wandb
# Disable wandb mặc định — không cần login
wandb.setup(wandb.Settings(mode="disabled"))

from config_schema import MainConfig
from surrogates import (
    ClipB16FeatureExtractor,
    ClipL336FeatureExtractor,
    ClipB32FeatureExtractor,
    ClipLaionFeatureExtractor,
    EnsembleFeatureLoss,
    EnsembleFeatureExtractor,
)
from utils import hash_training_config, setup_wandb, ensure_dir

BACKBONE_MAP: Dict[str, type] = {
    "L336": ClipL336FeatureExtractor,
    "B16": ClipB16FeatureExtractor,
    "B32": ClipB32FeatureExtractor,
    "Laion": ClipLaionFeatureExtractor,
}


# ---------------------------------------------------------------------------
# Dataset: numeric sort, hỗ trợ cả flat dir lẫn subdir
# ---------------------------------------------------------------------------

def _numeric_key(path: str) -> int:
    """Lấy số từ tên file. '42.png' -> 42, '0007.jpg' -> 7."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r'\d+', stem)
    return int(m.group()) if m else 0


class NumericOrderDataset(torch.utils.data.Dataset):
    """
    Dataset load ảnh từ thư mục và sort theo numeric order.
    Hỗ trợ:
      - Flat: root/*.png
      - Subdir: root/class/*.png  (dùng với bigscale_1000/nips17/)
    """
    VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def __init__(self, root_dir: str, transform=None):
        self.transform = transform
        self.samples: List[str] = []

        subdirs = [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith(".")
        ]

        if subdirs:
            for subdir in subdirs:
                for fname in os.listdir(subdir):
                    if os.path.splitext(fname)[1].lower() in self.VALID_EXT:
                        self.samples.append(os.path.join(subdir, fname))
        else:
            for fname in os.listdir(root_dir):
                if os.path.splitext(fname)[1].lower() in self.VALID_EXT:
                    self.samples.append(os.path.join(root_dir, fname))

        if not self.samples:
            raise RuntimeError(f"No images found in: {root_dir}")

        # Sort numeric
        self.samples.sort(key=_numeric_key)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, str]:
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0, path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_tensor(pic: Image.Image) -> torch.Tensor:
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    return img.permute((2, 0, 1)).contiguous().to(dtype=torch.get_default_dtype())


def set_environment(seed: int = 2023):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_str: str) -> str:
    """Fallback về MPS (Apple Silicon) hoặc CPU nếu CUDA không có."""
    if "cuda" in device_str:
        if torch.cuda.is_available():
            try:
                parts = device_str.split(":")
                if len(parts) > 1 and int(parts[1]) >= torch.cuda.device_count():
                    print(f"  [Device] '{device_str}' ngoài phạm vi, dùng cuda:0")
                    return "cuda:0"
            except ValueError:
                return "cuda:0"
            return device_str
        if torch.backends.mps.is_available():
            print("  [Device] Không có CUDA, dùng MPS (Apple Silicon)")
            return "mps"
        print("  [Device] Không có CUDA/MPS, dùng CPU")
        return "cpu"
    return device_str


def get_models(cfg: MainConfig):
    if not cfg.model.ensemble and len(cfg.model.backbone) > 1:
        raise ValueError("ensemble=False nhưng có nhiều hơn 1 backbone")
    models = []
    for name in cfg.model.backbone:
        if name not in BACKBONE_MAP:
            raise ValueError(f"Unknown backbone: {name}. Options: {list(BACKBONE_MAP)}")
        model = BACKBONE_MAP[name]().eval().to(cfg.model.device).requires_grad_(False)
        models.append(model)

    extractor = EnsembleFeatureExtractor(models) if cfg.model.ensemble else models[0]
    loss_fn = EnsembleFeatureLoss(models)
    return extractor, loss_fn


def log_metrics(pbar, metrics: dict, img_index: int, epoch: int = None):
    pbar.set_postfix({k: f"{v:.5f}" if "sim" in k else f"{v:.3f}" for k, v in metrics.items()})
    wm = {f"img{img_index}_{k}": v for k, v in metrics.items()}
    if epoch is not None:
        wm["epoch"] = epoch
    wandb.log(wm)


# ---------------------------------------------------------------------------
# Attack functions  (batch-aware)
# ---------------------------------------------------------------------------

def _compute_loss(cfg, extractor, loss_fn, adv_image, source_crop) -> Tuple[torch.Tensor, dict]:
    global_feats = extractor(adv_image)
    global_sim = loss_fn(global_feats)
    metrics = {"global_similarity": global_sim.item()}

    if cfg.model.use_source_crop:
        local_feats = extractor(source_crop(adv_image))
        local_sim = loss_fn(local_feats)
        metrics["local_similarity"] = local_sim.item()
        return local_sim, metrics
    return global_sim, metrics


def fgsm_attack(cfg, extractor, loss_fn, source_crop, target_crop,
                img_index, image_org, image_tgt) -> torch.Tensor:
    delta = torch.zeros_like(image_org, requires_grad=True)
    pbar = tqdm(range(cfg.optim.steps), desc="FGSM")

    for epoch in pbar:
        with torch.no_grad():
            loss_fn.set_ground_truth(target_crop(image_tgt))

        loss, metrics = _compute_loss(cfg, extractor, loss_fn, image_org + delta, source_crop)
        metrics.update({"max_delta": delta.abs().max().item(),
                        "mean_delta": delta.abs().mean().item()})
        log_metrics(pbar, metrics, img_index, epoch)

        grad = torch.autograd.grad(loss, delta)[0]
        delta.data = torch.clamp(delta + cfg.optim.alpha * grad.sign(),
                                 -cfg.optim.epsilon, cfg.optim.epsilon)

    return torch.clamp((image_org + delta) / 255.0, 0.0, 1.0)


def mifgsm_attack(cfg, extractor, loss_fn, source_crop, target_crop,
                  img_index, image_org, image_tgt) -> torch.Tensor:
    delta = torch.zeros_like(image_org, requires_grad=True)
    momentum = torch.zeros_like(image_org)
    pbar = tqdm(range(cfg.optim.steps), desc="MI-FGSM")

    for epoch in pbar:
        with torch.no_grad():
            loss_fn.set_ground_truth(target_crop(image_tgt))

        loss, metrics = _compute_loss(cfg, extractor, loss_fn, image_org + delta, source_crop)
        metrics.update({"max_delta": delta.abs().max().item(),
                        "mean_delta": delta.abs().mean().item()})
        log_metrics(pbar, metrics, img_index, epoch)

        grad = torch.autograd.grad(loss, delta)[0]
        momentum = momentum * 0.9 + grad
        delta.data = torch.clamp(delta + cfg.optim.alpha * momentum.sign(),
                                 -cfg.optim.epsilon, cfg.optim.epsilon)

    return torch.clamp((image_org + delta) / 255.0, 0.0, 1.0)


def pgd_attack(cfg, extractor, loss_fn, source_crop, target_crop,
               img_index, image_org, image_tgt) -> torch.Tensor:
    delta = torch.zeros_like(image_org, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=cfg.optim.alpha)
    pbar = tqdm(range(cfg.optim.steps), desc="PGD")

    for epoch in pbar:
        with torch.no_grad():
            loss_fn.set_ground_truth(target_crop(image_tgt))

        loss, metrics = _compute_loss(cfg, extractor, loss_fn, image_org + delta, source_crop)
        metrics.update({"max_delta": delta.abs().max().item(),
                        "mean_delta": delta.abs().mean().item()})
        log_metrics(pbar, metrics, img_index, epoch)

        optimizer.zero_grad()
        (-loss).backward()       # maximize similarity
        optimizer.step()
        delta.data = torch.clamp(delta, -cfg.optim.epsilon, cfg.optim.epsilon)

    return torch.clamp((image_org + delta) / 255.0, 0.0, 1.0)


ATTACK_FN = {"fgsm": fgsm_attack, "mifgsm": mifgsm_attack, "pgd": pgd_attack}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models")
def main(cfg: MainConfig):
    set_environment()
    setup_wandb(cfg, tags=["image_generation"])
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    extractor, loss_fn = get_models(cfg)

    transform_fn = transforms.Compose([
        transforms.Resize(cfg.model.input_res,
                          interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(cfg.model.input_res),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Lambda(to_tensor),
    ])

    clean_data  = NumericOrderDataset(cfg.data.cle_data_path, transform=transform_fn)
    target_data = NumericOrderDataset(cfg.data.tgt_data_path, transform=transform_fn)

    # Verify numeric order
    print("\n=== Verify order (first 5 clean / target) ===")
    for i in range(min(5, len(clean_data))):
        print(f"  [{i}] clean={os.path.basename(clean_data.samples[i])}  "
              f"target={os.path.basename(target_data.samples[i])}")
    print("=============================================\n")

    clean_loader  = torch.utils.data.DataLoader(clean_data,  batch_size=cfg.data.batch_size, shuffle=False)
    target_loader = torch.utils.data.DataLoader(target_data, batch_size=cfg.data.batch_size, shuffle=False)

    source_crop = (transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale)
                   if cfg.model.use_source_crop else nn.Identity())
    target_crop = (transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale)
                   if cfg.model.use_target_crop else nn.Identity())

    attack_fn    = ATTACK_FN[cfg.attack]
    config_hash  = hash_training_config(cfg)
    total_done   = 0
    total_batches = (cfg.data.num_samples + cfg.data.batch_size - 1) // cfg.data.batch_size

    for i, ((img_org, _, path_org), (img_tgt, _, path_tgt)) in enumerate(
            zip(clean_loader, target_loader)):

        if total_done >= cfg.data.num_samples:
            break

        print(f"\nBatch [{i+1}/{total_batches}]  "
              f"imgs {total_done}–{total_done + len(path_org) - 1}  "
              f"({os.path.basename(path_org[0])} … {os.path.basename(path_org[-1])})")

        img_org = img_org.to(cfg.model.device)
        img_tgt = img_tgt.to(cfg.model.device)

        adv_images = attack_fn(
            cfg=cfg,
            extractor=extractor,
            loss_fn=loss_fn,
            source_crop=source_crop,
            target_crop=target_crop,
            img_index=i,
            image_org=img_org,
            image_tgt=img_tgt,
        )

        # Save each image in the batch
        for b in range(len(path_org)):
            folder = os.path.basename(os.path.dirname(path_org[b]))
            stem   = os.path.splitext(os.path.basename(path_org[b]))[0]
            out_dir = os.path.join(cfg.data.output, "img", config_hash, folder)
            ensure_dir(out_dir)
            save_path = os.path.join(out_dir, stem + ".png")
            torchvision.utils.save_image(adv_images[b], save_path)

        total_done += len(path_org)

    wandb.finish()
    print(f"\nDone. Saved {total_done} adversarial images to {cfg.data.output}/img/{config_hash}/")


if __name__ == "__main__":
    main()
