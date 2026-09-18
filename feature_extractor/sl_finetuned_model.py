import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import AdamW
import torchvision
from torch.utils.data import DataLoader

from PIL import ImageFilter, ImageOps, Image
from torchvision import transforms

from peft import LoraConfig, get_peft_model, BOFTConfig
from transformers import AutoModel
from peft.peft_model import PeftModel
from peft.config import PeftConfig

from tqdm import tqdm
import copy

from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import vision_transformer as vits
from vision_transformer import DINOHead

import utils

import argparse
import os

torch.manual_seed(42)  # Set random seed for reproducibility

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        self.backbone = backbone

        self.head = head

    def forward(self, x):
        # Run the head forward on the concatenated features.
        pooler_output = self.backbone(x).pooler_output

        cls_output = self.head(pooler_output)

        return cls_output, pooler_output

def nt_xent(z, tau):
    """NT-Xent (SimCLR Eq. 1). z: (2N, D), rows 0..N-1 = view 1, N..2N-1 = view 2, same order.

    Paper Eq. 2 is underspecified (no outer sum, unclear denominator and anchor set), see Q6.
    Here: 2N anchors, symmetric, denominator = all other 2N-1 views, positive included.
    """
    n = z.shape[0] // 2
    z = F.normalize(z, dim=1)
    sim = (z @ z.t()) / tau
    sim.fill_diagonal_(float("-inf"))          # a view is never its own negative
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def finetune_dino(train_set, num_classes, model_name="facebook/dino-vitb16"):
        
    interpolation = 3
    crop_pct = 0.875
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image_size = 224
    transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                        mean=torch.tensor(mean),
                        std=torch.tensor(std))
        ])

    batch_size = int(os.environ.get("FT_BATCH_SIZE", 128))

    epochs = int(os.environ.get("FT_EPOCHS", 10))

    lora_model = load_model(num_classes, model_name)

    # Optimizer
    optimizer = AdamW(lora_model.parameters(), lr=1e-3)

    lora_model.to(device)

    print(train_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    # Training loop
    accum = int(os.environ.get("FT_ACCUM", 1))
    lam = float(os.environ.get("FT_LAMBDA", 0))    # Eq. 3; 0 = upstream (CE only)
    tau = float(os.environ.get("FT_TAU", 0.2))
    for epoch in range(epochs):  # Adjust epochs as needed
        optimizer.zero_grad()
        pending, running, running_con, nb = 0, 0.0, 0.0, 0

        for bidx, batch in tqdm(enumerate(train_loader)):

            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            if lam > 0:
                # one forward over 2N images: rows 0..N-1 = view 1, N..2N-1 = view 2
                images = torch.cat([images, batch["images2"].to(device)], dim=0)
                output, pooler_output = lora_model(images)
                cls_loss = F.cross_entropy(output[:labels.shape[0]], labels)   # CE on view 1
                con_loss = nt_xent(pooler_output, tau)
                loss = ((1 - lam) * cls_loss + lam * con_loss) / accum
            else:
                output, pooler_output = lora_model(images)
                cls_loss = F.cross_entropy(output, labels)
                con_loss = None
                loss = cls_loss / accum

            # Backward pass and optimization
            loss.backward()
            pending += 1

            if pending == accum:
                optimizer.step()
                optimizer.zero_grad()
                pending = 0

            running += cls_loss.item(); nb += 1
            if con_loss is not None:
                running_con += con_loss.item()

        if pending > 0:          # flush phan du cuoi epoch
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch}: mean CE = {running / max(nb, 1):.4f} "
              f"mean CON = {running_con / max(nb, 1):.4f} "
              f"(lambda={lam} tau={tau} batch={batch_size} accum={accum} "
              f"effective={batch_size * accum} views={2 if lam > 0 else 1})")

        # PATCH 5: path from FT_CKPT_PATH (default unchanged); also keep the frozen, randomly initialised HF pooler
        ckpt_path = os.environ.get("FT_CKPT_PATH", "/tmp/ft_ckpt.pt")
        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
        ckpt = {n: p.detach().cpu() for n, p in lora_model.named_parameters()
                if p.requires_grad or "pooler.dense" in n}
        torch.save({"epoch": epoch, "state": ckpt}, ckpt_path)

    return lora_model.backbone


def load_model(num_classes, model_name):

    if model_name == "dinov2_vitb14":
        model_name = "facebook/dinov2-base"

    if model_name == "dino_vitb16":
        model_name = "facebook/dino-vitb16"

    model = AutoModel.from_pretrained(model_name)


    peft_config = BOFTConfig(
        boft_block_size=4,
        boft_n_butterfly_factor=2,
        target_modules=["output.dense", "mlp.fc1", "mlp.fc2"],
        boft_dropout=0.1,
        bias="boft_only",
    )

    backbone = get_peft_model(model, peft_config)

    embed_dim = backbone.config.hidden_size

    lora_model = MultiCropWrapper(backbone, torch.nn.Linear(embed_dim, num_classes))

    lora_model.train()

    backbone.print_trainable_parameters()

    return lora_model
