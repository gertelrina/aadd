"""v3 frame-level models.

* :class:`WhisperFrameModel` — Whisper encoder (small / base, Apache 2.0) on 8-s
  windows with a per-20-ms multi-label head.
* :class:`CRNNFrameModel` — ~1 M-parameter CRNN on 64-band log-mel for CPU use.

Both take raw 16 kHz waveforms ``[B, N]`` (N = 8 s * 16000) and return logits
``[B, N // 320, n_classes]``; feature extraction runs inside the model on the
same device.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from aadd.v3.synth import CLASSES_V3

SR = 16000
HOP = 160


class WhisperLogMel(nn.Module):
    """Whisper's log-mel front-end in torch (n_fft 400, hop 160, 80 slaney mels, max-8 dB floor)."""

    def __init__(self, n_mels: int = 80):
        super().__init__()
        import librosa

        self.register_buffer("window", torch.hann_window(400), persistent=False)
        fb = torch.tensor(librosa.filters.mel(sr=SR, n_fft=400, n_mels=n_mels), dtype=torch.float32)
        self.register_buffer("fb", fb, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        st = torch.stft(x.float(), 400, HOP, window=self.window, return_complex=True)
        mag = st[..., :-1].abs() ** 2  # drop the last frame like Whisper: N/160 frames
        mel = self.fb @ mag
        logm = torch.clamp(mel, min=1e-10).log10()
        logm = torch.maximum(logm, logm.amax(dim=(-2, -1), keepdim=True) - 8.0)
        return (logm + 4.0) / 4.0  # [B, 80, T]


class WhisperFrameModel(nn.Module):
    def __init__(
        self,
        backbone: str = "openai/whisper-small",
        n_classes: int = len(CLASSES_V3),
        seconds: float = 8.0,
        pretrained: bool = True,
        head_dim: int = 256,
    ):
        super().__init__()
        from transformers import WhisperConfig, WhisperModel

        n_frames = int(seconds * SR / HOP)  # mel frames
        self.n_pos = n_frames // 2  # encoder frames (20 ms)
        if pretrained:
            enc = WhisperModel.from_pretrained(backbone).encoder
        else:
            enc = WhisperModel(WhisperConfig.from_pretrained(backbone)).encoder
        old = enc.embed_positions.weight.data
        emb = nn.Embedding(self.n_pos, old.shape[1])
        emb.weight.data.copy_(old[: self.n_pos])
        emb.weight.requires_grad_(False)
        enc.embed_positions = emb
        enc.config.max_source_positions = self.n_pos
        self.encoder = enc
        self.backbone = backbone
        d = enc.config.d_model
        L = enc.config.encoder_layers + 1
        self.frontend = WhisperLogMel(enc.config.num_mel_bins)
        self.layer_w = nn.Parameter(torch.zeros(L))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Conv1d(d, head_dim, 5, padding=2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(head_dim, head_dim, 5, padding=2),
            nn.GELU(),
        )
        self.out = nn.Linear(head_dim, n_classes)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        feats = self.frontend(wav)
        hs = self.encoder(feats, output_hidden_states=True).hidden_states  # L x [B, T, d]
        w = torch.softmax(self.layer_w, 0)
        h = sum(w[i] * hs[i] for i in range(len(hs)))
        h = self.norm(h).transpose(1, 2)
        h = self.head(h).transpose(1, 2)
        return self.out(h)  # [B, T=400, C]


class CRNNFrameModel(nn.Module):
    """Small CPU model: log-mel 64 x T -> conv (freq pooling, one time pooling) -> BiGRU -> per-20-ms logits."""

    def __init__(self, n_classes: int = len(CLASSES_V3), n_mels: int = 64, width: int = 64, rnn: int = 128):
        super().__init__()
        import librosa

        self.register_buffer("window", torch.hann_window(512), persistent=False)
        self.register_buffer(
            "fb",
            torch.tensor(librosa.filters.mel(sr=SR, n_fft=512, n_mels=n_mels), dtype=torch.float32),
            persistent=False,
        )

        def blk(ci, co, pool):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.GELU(),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.GELU(),
                nn.MaxPool2d(pool),
            )

        self.cnn = nn.Sequential(
            blk(1, width, (2, 2)),
            blk(width, width * 2, (2, 1)),
            blk(width * 2, width * 2, (4, 1)),
            blk(width * 2, width * 2, (4, 1)),
        )
        self.rnn = nn.GRU(
            width * 2 * (n_mels // 64), rnn, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1
        )
        self.out = nn.Linear(2 * rnn, n_classes)

    def features(self, wav: torch.Tensor) -> torch.Tensor:
        st = torch.stft(wav.float(), 512, HOP, window=self.window, return_complex=True)[..., :-1]
        mel = self.fb @ (st.abs() ** 2)
        # absolute dB (re. full scale), so level-dependent cues survive
        db = 10 * torch.log10(mel + 1e-10)
        return ((db + 100.0) / 100.0).clamp(0, 1.5).unsqueeze(1)  # [B, 1, M, T]

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.cnn(self.features(wav))  # [B, Cc, M', T/2]
        b, c, m, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * m)
        x, _ = self.rnn(x)
        return self.out(x)


def build(arch: str, **kw) -> nn.Module:
    if arch == "whisper-small":
        return WhisperFrameModel("openai/whisper-small", **kw)
    if arch == "whisper-base":
        return WhisperFrameModel("openai/whisper-base", **kw)
    if arch == "crnn":
        return CRNNFrameModel(**kw)
    raise ValueError(arch)


def masked_bce(
    logits: torch.Tensor, target: torch.Tensor, ignore: torch.Tensor, pos_weight: torch.Tensor | None = None
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none", pos_weight=pos_weight)
    keep = (~ignore).float()
    return (loss * keep).sum() / keep.sum().clamp(min=1.0)
