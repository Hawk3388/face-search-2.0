"""Train on prepared face crops: python -m model2.train --dataset model2/prepared."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model2.data import create_pairs, load_people, split_people
from model2.network import SiameseNetwork, inference_transform
from face_search.evaluation import calibrate_threshold, verification_metrics

PREPROCESSING = "face-crop-rgb224-imagenet-v1"


class PersonPairDataset(Dataset):
    def __init__(self, people, transform=None, pairs_per_person=20, seed=42):
        self.people = people
        self.transform = transform or inference_transform()
        self.pairs_per_person = pairs_per_person
        self.seed = seed
        self.resample(0)

    def resample(self, epoch):
        self.pairs = create_pairs(self.people, self.pairs_per_person, self.seed + epoch)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        first, second, label = self.pairs[index]
        with Image.open(first) as image:
            a = self.transform(image.convert("RGB"))
        with Image.open(second) as image:
            b = self.transform(image.convert("RGB"))
        return a, b, torch.tensor(label, dtype=torch.float32)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, similarity, labels):
        distance = 1 - similarity
        return (labels * distance.square() + (1 - labels) * (self.margin - distance).clamp(min=0).square()).mean()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, count = 0.0, 0
    for first, second, labels in loader:
        first, second, labels = first.to(device), second.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(first, second), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
        count += len(labels)
    return total / count


def validate(model, loader, criterion, device):
    model.eval()
    total, count, distances, targets = 0.0, 0, [], []
    with torch.inference_mode():
        for first, second, labels in loader:
            labels = labels.to(device)
            similarity = model(first.to(device), second.to(device))
            total += criterion(similarity, labels).item() * len(labels)
            count += len(labels)
            distances.extend((1 - similarity).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    return total / count, np.asarray(distances), np.asarray(targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("model2/prepared"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--target-far", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive")
    manifest = args.dataset / "preprocessing.json"
    if not manifest.exists() or json.loads(manifest.read_text())["preprocessing"] != PREPROCESSING:
        parser.error("Prepare face crops first: python -m model2.prepare_dataset --input ... --output ...")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    people = load_people(args.dataset)
    # Retain a person-disjoint test split; never choose checkpoints on test people.
    if len(people) < 6:
        parser.error("Training with independent train/validation/test splits needs at least six people")
    train_val, test_people = split_people(people, 0.2, args.seed)
    train_people, val_people = split_people(train_val, 0.2, args.seed + 1)
    split = {"train": sorted(train_people), "validation": sorted(val_people), "test": sorted(test_people)}
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if checkpoint and (checkpoint["split"] != split or checkpoint["seed"] != args.seed):
        parser.error("Resume requires the same dataset identity split and seed")
    augmentation = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_data = PersonPairDataset(train_people, augmentation, 50, args.seed)
    val_data = PersonPairDataset(val_people, inference_transform(), 50, args.seed + 1)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers)
    model = SiameseNetwork(pretrained=checkpoint is None).to(device)
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    for layer in (model.encoder.layer3, model.encoder.layer4):
        for parameter in layer.parameters():
            parameter.requires_grad = True
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    criterion = ContrastiveLoss()
    start, best_loss, stale = 0, float("inf"), 0
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start, best_loss, stale = checkpoint["epoch"] + 1, checkpoint["best_loss"], checkpoint["stale"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    output = args.output or Path("runs") / datetime.now().strftime("siamese-%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    if args.resume and (args.resume.parent / "best.pth").exists():
        shutil.copy2(args.resume.parent / "best.pth", output / "best.pth")
    (output / "split.json").write_text(json.dumps(split, indent=2))
    print(f"Output: {output}; train={len(train_people)}, validation={len(val_people)}, test={len(test_people)}")
    for epoch in range(start, args.epochs):
        train_data.resample(epoch)
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, distances, labels = validate(model, val_loader, criterion, device)
        threshold = calibrate_threshold(distances, labels, args.target_far)
        metrics = verification_metrics(distances, labels, threshold)
        scheduler.step(val_loss)
        improved = val_loss < best_loss
        best_loss, stale = (val_loss, 0) if improved else (best_loss, stale + 1)
        saved = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "stale": stale,
            "embedding_dim": 512,
            "preprocessing": PREPROCESSING,
            "split": split,
            "seed": args.seed,
            "threshold": threshold,
            "metric": "cosine",
            "validation": metrics,
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }
        for filename in ["last.pth", "best.pth"] if improved else ["last.pth"]:
            temporary = output / (filename + ".tmp")
            torch.save(saved, temporary)
            temporary.replace(output / filename)
        print(
            f"Epoch {epoch + 1}: train={train_loss:.4f} val={val_loss:.4f} FAR={metrics['far']:.3f} FRR={metrics['frr']:.3f}"
        )
        if stale >= args.patience:
            break


if __name__ == "__main__":
    main()
