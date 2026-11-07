"""Model construction, checkpoint format and the built-in model registry.

Checkpoint format (``*.pt``, ``torch.save`` of a dict)::

    {
      "format": "aadd-checkpoint-1",
      "model_state_dict": {...},
      "arch": "efficientnet_b0",
      "in_channels": 3,
      "classes": [...],
      "features": FeatureSpec.to_dict(),
      "meta": {"name": ..., "trained_on": ..., "epoch": ..., "metrics": {...}},
    }

The v1 checkpoint (originally ``efficientnet_b0_sep2_mels_2025-03-11_12-01-06.pth``,
which also contained the Adam state) was converted to this format with
``aadd convert-checkpoint``; the weights are unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torchvision import models

from aadd.constants import CLASSES
from aadd.features import FeatureSpec

CHECKPOINT_FORMAT = "aadd-checkpoint-1"
# Weights are hosted on the Hugging Face Hub (agertel/aadd) and cached in CACHE_DIR on first use.
# A local ``weights/`` folder next to the package, if present, takes precedence (for development).
HF_REPO = "agertel/aadd"
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

# name -> (file name, description). Files live in <repo>/weights.
REGISTRY = {
    "v1": {
        "file": "aadd_v1_efficientnet_b0_png.pt",
        "url": "https://huggingface.co/agertel/aadd/resolve/main/aadd_v1_efficientnet_b0_png.pt",
        "description": "First model (2025-03-11): EfficientNet-B0 on matplotlib-rendered mel PNGs, 8 outputs.",
    },
    "v2": {
        "file": "aadd_v2_efficientnet_b0_mel.pt",
        "url": "https://huggingface.co/agertel/aadd/resolve/main/aadd_v2_efficientnet_b0_mel.pt",
        "description": "EfficientNet-B0 on direct log-mel tensors (1x128x63), same labels, 3-5x faster than v1.",
    },
    "v3": {
        "file": "aadd_v3_1_whisper_small_frames.pt",
        "url": "https://huggingface.co/agertel/aadd/resolve/main/aadd_v3_1_whisper_small_frames.pt",
        "description": "v3.1: Whisper-small encoder, 12 realistic classes per 20 ms; fewer music false alarms on real babble.",
    },
    "v3.0": {
        "file": "aadd_v3_whisper_small_frames.pt",
        "url": "https://huggingface.co/agertel/aadd/resolve/main/aadd_v3_whisper_small_frames.pt",
        "description": "First v3 release (2026-11-05); superseded by v3.1.",
    },
    "v3-lite": {
        "file": "aadd_v3_lite_crnn_frames.pt",
        "description": "1.3M-parameter CRNN distilled from v3: same 12 classes per 20 ms frame, fast on CPU.",
    },
}
CACHE_DIR = Path(os.environ.get("AADD_CACHE", Path.home() / ".cache" / "aadd"))
DEFAULT_MODEL = "v3"


def build_model(
    arch: str = "efficientnet_b0",
    num_classes: int = len(CLASSES),
    in_channels: int = 3,
    pretrained: bool = True,
) -> nn.Module:
    """Build a classifier with ``num_classes`` logits (sigmoid applied outside).

    ``in_channels != 3`` replaces the stem convolution; when pretrained, the
    RGB kernels are summed so the 1-channel stem starts from the ImageNet
    features (the usual trick for grayscale inputs).
    """
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        if in_channels != 3:
            old = m.features[0][0]
            new = nn.Conv2d(
                in_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None
            )
            with torch.no_grad():
                if pretrained:
                    w = old.weight.sum(dim=1, keepdim=True)
                    new.weight.copy_(w.repeat(1, in_channels, 1, 1) / in_channels)
            m.features[0][0] = new
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    if arch == "small_cnn":
        return SmallCNN(num_classes=num_classes, in_channels=in_channels)
    raise ValueError(f"unknown arch {arch!r}")


class SmallCNN(nn.Module):
    """A ~0.3M-parameter CNN for 1xF xT log-mel inputs; a cheap CPU-friendly baseline."""

    def __init__(self, num_classes: int, in_channels: int = 1, width: int = 32):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, width), block(width, width * 2), block(width * 2, width * 4), block(width * 4, width * 4)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(width * 4, num_classes)
        )

    def forward(self, x):
        return self.head(self.features(x))


def save_checkpoint(
    path: str | os.PathLike,
    model: nn.Module,
    arch: str,
    in_channels: int,
    features: FeatureSpec,
    classes: list[str] = CLASSES,
    meta: dict | None = None,
) -> None:
    ckpt = {
        "format": CHECKPOINT_FORMAT,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "arch": arch,
        "in_channels": in_channels,
        "classes": list(classes),
        "features": features.to_dict(),
        "meta": meta or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def resolve_model_path(name_or_path: str) -> Path:
    if name_or_path in REGISTRY:
        spec = REGISTRY[name_or_path]
        for d in (WEIGHTS_DIR, CACHE_DIR):
            if (d / spec["file"]).exists():
                return d / spec["file"]
        if spec.get("url"):
            return _download(spec["url"], CACHE_DIR / spec["file"])
        raise FileNotFoundError(f"no local file and no download URL for model {name_or_path!r}")
    p = Path(name_or_path)
    if p.exists():
        return p
    raise FileNotFoundError(f"model {name_or_path!r} is neither a registry name {list(REGISTRY)} nor an existing file")


def cached_path(name: str) -> Path | None:
    """Local path of a registry model if it is already on disk, else None (no download)."""
    spec = REGISTRY[name]
    for d in (WEIGHTS_DIR, CACHE_DIR):
        if (d / spec["file"]).exists():
            return d / spec["file"]
    return None


def _download(url: str, dst: Path) -> Path:
    import urllib.request

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    print(f"downloading {url} -> {dst}", flush=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dst)
    return dst


def checkpoint_format(name_or_path: str) -> str:
    ck = torch.load(resolve_model_path(name_or_path), map_location="cpu", weights_only=False)
    return ck.get("format", "legacy")


def load_checkpoint(name_or_path: str, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Return ``(model.eval(), checkpoint_dict)``.

    Also accepts the raw legacy v1 checkpoint (``{'model_state_dict': ..., 'optimizer_state_dict': ...}``),
    which is interpreted as a v1 'png' EfficientNet-B0.
    """
    path = resolve_model_path(name_or_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        ckpt = _upgrade_legacy_checkpoint(ckpt, source=str(path))
    model = build_model(
        ckpt["arch"], num_classes=len(ckpt["classes"]), in_channels=ckpt["in_channels"], pretrained=False
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def _upgrade_legacy_checkpoint(ckpt: dict, source: str = "") -> dict:
    sd = ckpt.get("model_state_dict", ckpt)
    n_out = sd["classifier.1.weight"].shape[0]
    if n_out != len(CLASSES):
        raise ValueError(f"legacy checkpoint has {n_out} outputs, expected {len(CLASSES)}")
    return {
        "format": CHECKPOINT_FORMAT,
        "model_state_dict": sd,
        "arch": "efficientnet_b0",
        "in_channels": 3,
        "classes": list(CLASSES),
        "features": FeatureSpec(kind="png").to_dict(),
        "meta": {
            "name": "v1",
            "source": source,
            "epoch": ckpt.get("epoch"),
            "val_accuracy_by_one": ckpt.get("val_accuracy_by_one"),
            "loss": ckpt.get("loss"),
        },
    }


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
