"""Qwen3-VL as a yes/no relevance scorer over cutouts (dev-only).

The retrieval side of this pipeline is a bi-encoder: one text embedding against a
million cutout embeddings. This module is the cross-encoder half of the usual
retrieve-then-verify shape — the VLM sees the query and one cutout together, which the
bi-encoder never does.

**Read the probability, not the answer.** The score is `p("yes")` renormalised over the
two answer tokens at the first generated position, taken straight from the logits with
no sampling. Two reasons, both measured elsewhere and both load-bearing:

- VLMs answer "yes" to object-existence questions far more often than they should
  (POPE, arXiv 2305.10355), so the *decision* is badly calibrated even where the
  *ranking* is informative. A caller that thresholds the label inherits that bias; a
  caller that ranks on the probability does not;
- a per-query offset in the score is not neutral downstream. The ratio score divides by
  the query's best cluster, so a constant shifts every ratio. Callers that
  mix this into a similarity should rescale it per query, exactly as
  `candidates.normalize_prototype_similarities` does for the prototype columns.

Loading is 4-bit NF4 by default: the dev box has 8 GB of VRAM and the mirrored online
service is usually holding ~4 GB of it, so bf16 does not fit next to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from toolbox.logging import logger

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_BATCH_SIZE = 8
# Cutouts are 224x224; this only bounds the pathological case of a caller passing a
# full ERP, which would blow the vision-token budget and the VRAM with it.
DEFAULT_MAX_PIXELS = 512 * 512
# One sentence, one question, no chain of thought: the score comes from the first
# answer token, so anything the model would say before it is wasted computation.
DEFAULT_QUESTION = (
    "Is this a photo of {query}? Look only at the object in the centre of the image. "
    "Answer with a single word, yes or no."
)


@dataclass(frozen=True)
class GateConfig:
    """Everything that changes a score, in one hashable object.

    Carried around so a benchmark row can record exactly what produced it: the model,
    the question and the quantization all move the numbers.
    """

    model_id: str = DEFAULT_MODEL_ID
    question: str = DEFAULT_QUESTION
    quantization: str = "4bit"
    batch_size: int = DEFAULT_BATCH_SIZE
    max_pixels: int = DEFAULT_MAX_PIXELS


class VlmYesNoScorer:
    """Score (cutout, query) pairs by the probability of the answer "yes".

    The model is loaded on first use, not in `__init__`, so constructing one in a
    benchmark that ends up not needing it costs nothing.
    """

    def __init__(self, config: GateConfig | None = None) -> None:
        self.config = config or GateConfig()
        self._model = None
        self._processor = None
        self._yes_ids: list[int] = []
        self._no_ids: list[int] = []

    def _load(self) -> None:
        """Load the model, processor and the answer-token ids, once."""
        if self._model is not None:
            return
        # Imported here rather than at module scope: transformers pulls in torch and
        # several seconds of CUDA init, and most of this repo never touches either.
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen3VLForConditionalGeneration,
        )

        quantization_config = None
        dtype = torch.bfloat16
        if self.config.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        logger.info(
            "Loading %s (%s) for yes/no gating",
            self.config.model_id,
            self.config.quantization,
        )
        processor = AutoProcessor.from_pretrained(
            self.config.model_id, max_pixels=self.config.max_pixels
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_id,
            dtype=dtype,
            quantization_config=quantization_config,
            device_map="cuda:0",
        )
        model.eval()
        tokenizer = processor.tokenizer
        # Both cases and the leading-space variants: which one the template makes
        # first differs between models, and missing it would read as "always no".
        self._yes_ids = _answer_token_ids(tokenizer, ("yes", "Yes", " yes", " Yes"))
        self._no_ids = _answer_token_ids(tokenizer, ("no", "No", " no", " No"))
        if not self._yes_ids or not self._no_ids:
            raise RuntimeError("Could not resolve single-token yes/no answers")
        self._processor = processor
        self._model = model

    def score_images(self, images: list[Image.Image], query: str) -> np.ndarray:
        """Return `p(yes)` for each image against one query.

        Args:
            images: Cutouts, already loaded and RGB.
            query: The search prompt, inserted into the configured question.

        Returns:
            Probabilities in `[0, 1]`, aligned with ``images``.
        """
        if not images:
            return np.empty(0, dtype=np.float64)
        self._load()
        import torch

        assert self._model is not None and self._processor is not None  # noqa: S101
        question = self.config.question.format(query=query)
        scores: list[float] = []
        for start in range(0, len(images), self.config.batch_size):
            batch = images[start : start + self.config.batch_size]
            texts = [
                self._processor.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": question},
                            ],
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for _ in batch
            ]
            inputs = self._processor(
                text=texts, images=batch, padding=True, return_tensors="pt"
            ).to(self._model.device)
            with torch.inference_mode():
                logits = self._model(**inputs).logits[:, -1, :].float()
            yes = torch.logsumexp(logits[:, self._yes_ids], dim=-1)
            no = torch.logsumexp(logits[:, self._no_ids], dim=-1)
            # Renormalised over the two answers only: the mass the model puts on any
            # other continuation says nothing about the question that was asked.
            probability = torch.sigmoid(yes - no)
            scores.extend(probability.detach().cpu().numpy().tolist())
        return np.asarray(scores, dtype=np.float64)

    def score_paths(self, paths: list[Path], query: str) -> np.ndarray:
        """`score_images` for cutouts on disk; unreadable files score `NaN`.

        A missing thumbnail is not zero evidence dressed as a rejection — the caller
        has to be able to tell "the model said no" from "there was nothing to show it".
        """
        images: list[Image.Image] = []
        usable: list[int] = []
        for index, path in enumerate(paths):
            try:
                with Image.open(path) as handle:
                    images.append(handle.convert("RGB"))
                usable.append(index)
            except (OSError, ValueError):
                logger.debug("Unreadable cutout %s", path)
        scores = np.full(len(paths), np.nan, dtype=np.float64)
        if images:
            scores[usable] = self.score_images(images, query)
        return scores


def _answer_token_ids(tokenizer: Any, variants: tuple[str, ...]) -> list[int]:
    """Ids of the single-token spellings of an answer word."""
    ids: list[int] = []
    for variant in variants:
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if len(encoded) == 1:
            ids.append(int(encoded[0]))
    return sorted(set(ids))
