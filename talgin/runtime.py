from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import torch
from tokenizers import Tokenizer

from .model import ModelConfig, TalginDecoderModel


SPECIAL = {"pad": 0, "bos": 2, "eos": 3, "system": 4, "user": 5, "assistant": 6}
TRAINED_CONTEXT = 2048
DEFAULT_TOKENIZER_SHA256 = "17505590a7e2631989c4b05196577a9875ba14c93c6b12a9266daedb54550f09"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class TalginRuntime:
    def __init__(self):
        checkpoint_raw = os.getenv("TALGIN_CHECKPOINT_PATH", "").strip()
        tokenizer_raw = os.getenv("TALGIN_TOKENIZER_PATH", "").strip()
        self.checkpoint_path = Path(checkpoint_raw).expanduser() if checkpoint_raw else None
        self.tokenizer_path = Path(tokenizer_raw).expanduser() if tokenizer_raw else None
        self.expected_checkpoint_sha = os.getenv("TALGIN_CHECKPOINT_SHA256", "").strip().lower()
        self.expected_tokenizer_sha = os.getenv(
            "TALGIN_TOKENIZER_SHA256", DEFAULT_TOKENIZER_SHA256
        ).strip().lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[TalginDecoderModel] = None
        self.tokenizer: Optional[Tokenizer] = None
        self.checkpoint_sha: Optional[str] = None
        self.tokenizer_sha: Optional[str] = None
        self.error: Optional[str] = None
        self._load_if_configured()

    @property
    def available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _load_if_configured(self):
        if self.checkpoint_path is None or self.tokenizer_path is None:
            return
        if not self.checkpoint_path.is_file() or not self.tokenizer_path.is_file():
            self.error = "Configured checkpoint/tokenizer file is missing."
            return
        try:
            self.checkpoint_sha = sha256(self.checkpoint_path)
            self.tokenizer_sha = sha256(self.tokenizer_path)
            if self.expected_checkpoint_sha and self.checkpoint_sha != self.expected_checkpoint_sha:
                raise ValueError("Checkpoint SHA-256 mismatch")
            if self.expected_tokenizer_sha and self.tokenizer_sha != self.expected_tokenizer_sha:
                raise ValueError("Tokenizer SHA-256 mismatch")

            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            cfg = ModelConfig.from_dict(checkpoint["model_config"])
            model = TalginDecoderModel(cfg)
            model.load_state_dict(checkpoint["model"], strict=True)
            if not all(torch.isfinite(p).all().item() for p in model.parameters()):
                raise ValueError("Checkpoint contains non-finite weights")

            tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
            for name, token_id in SPECIAL.items():
                if tokenizer.encode(f"<|{name}|>", add_special_tokens=False).ids != [token_id]:
                    raise ValueError(f"Special token mismatch: {name}")

            self.model = model.to(self.device).eval()
            self.tokenizer = tokenizer
        except Exception as exc:
            self.error = str(exc)
            self.model = None
            self.tokenizer = None

    def info(self) -> dict:
        cfg = self.model.config_dict() if self.model is not None else ModelConfig().__dict__
        return {
            "name": "TalginAI V4",
            "available": self.available,
            "device": str(self.device),
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "parameters": self.model.parameter_count() if self.model is not None else 66_207_360,
            "architecture": {
                "type": "decoder-only-transformer",
                "gqa": True,
                "rope": True,
                "rmsnorm": True,
                "swiglu": True,
                "config": cfg,
            },
            "trained_context": TRAINED_CONTEXT,
            "checkpoint_sha256": self.checkpoint_sha,
            "tokenizer_sha256": self.tokenizer_sha,
            "error": self.error,
        }

    def _prompt_ids(self, prompt: str) -> list[int]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded")
        clean = prompt.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        part = self.tokenizer.encode("\n" + clean, add_special_tokens=False).ids
        return [SPECIAL["bos"], SPECIAL["user"], *part, SPECIAL["assistant"]]

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 64) -> dict:
        if not self.available:
            raise RuntimeError(self.error or "TalginAI model artifacts are not mounted")
        if not 1 <= max_new_tokens <= 256:
            raise ValueError("max_new_tokens must be 1..256")
        ids = self._prompt_ids(prompt)
        if len(ids) + max_new_tokens > TRAINED_CONTEXT:
            raise ValueError("prompt plus generation exceeds trained context")

        x = torch.tensor([ids], device=self.device)
        output: list[int] = []
        eos = False
        for _ in range(max_new_tokens):
            logits, _ = self.model(x)
            scores = logits[0, -1].float().clone()
            scores[[SPECIAL["pad"], SPECIAL["bos"], SPECIAL["system"], SPECIAL["user"], SPECIAL["assistant"]]] = -float("inf")
            token = int(scores.argmax())
            if token == SPECIAL["eos"]:
                eos = True
                break
            output.append(token)
            x = torch.cat((x, torch.tensor([[token]], device=self.device)), dim=1)

        text = self.tokenizer.decode(output, skip_special_tokens=True).lstrip("\n")
        return {"text": text, "generated_tokens": len(output), "eos": eos, "device": str(self.device)}
