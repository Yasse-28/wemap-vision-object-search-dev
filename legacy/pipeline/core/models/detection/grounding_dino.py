from __future__ import annotations

import time
from typing import Any, List, Optional

import torch
import torch._inductor.config as ic
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from pipeline.core.detectors import Detection
from pipeline.core.logging import logger
from pipeline.core.models.base_model import SingletonModel

ic.fx_graph_cache = True


class GroundingDINOModel(SingletonModel):
    default_conf = {
        "name": "grounding_dino",
        "pretrained_model_name_or_path": "IDEA-Research/grounding-dino-tiny",
        "threshold": 0.06,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "compile": torch.cuda.is_available(),
        "max_length": 256,
        "batch_size": 1,
        "prompt": None,
    }

    def __init__(self, **conf: Any):
        super().__init__(**conf)
        logger.info("GroundingDINO loading model")
        start_time = time.time()
        self.device = self.conf.device
        model_path = self.conf.pretrained_model_name_or_path
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                model_path, dtype=torch.float16, low_cpu_mem_usage=True
            )
            .eval()
            .to(self.device)
        )
        self.prompt = self.conf.prompt
        self.text_inputs: Any = None
        if self.prompt is not None:
            self.text_inputs = self.processor.tokenizer(
                [self.prompt] * self.conf.batch_size,
                padding="max_length",
                max_length=self.conf.max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
        if self.conf.compile:
            self.model = torch.compile(self.model)
        logger.info("GroundingDINO model loaded in %.2fs", time.time() - start_time)

    def eval(self) -> "GroundingDINOModel":
        self.model.eval()
        return self

    def to(self, device: Any) -> "GroundingDINOModel":
        self.device = device
        self.model.to(device)
        return self

    @torch.inference_mode()
    def detect(
        self, images: List[Image.Image], text_prompt: Optional[str] = None
    ) -> List[List[Detection]]:
        if not images:
            return []

        detections_list = []
        # Prepare text inputs
        text_inputs = self.text_inputs
        if text_prompt is not None and isinstance(text_prompt, str):
            logger.debug("Using text prompt: %s", text_prompt)
            batch_prompts = [text_prompt] * self.conf.batch_size
            text_inputs = self.processor.tokenizer(
                batch_prompts,
                padding="max_length",
                max_length=self.conf.max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
        elif self.text_inputs is None and text_prompt is None:
            raise ValueError(
                "text_prompt was not preset and not input to the function. "
                "Please set the text_prompt in the constructor or input it "
                "to the detect function."
            )
        batch_size = self.conf.batch_size
        n = len(images)
        idx = 0
        while idx < n:
            batch_imgs = images[idx : idx + batch_size]
            num_additional_dummy_images = batch_size - len(batch_imgs)
            if len(batch_imgs) < batch_size:
                batch_imgs.extend(
                    [Image.new("RGB", (512, 512), color="white")]
                    * num_additional_dummy_images
                )
            image_inputs = self.processor.image_processor(
                images=batch_imgs, return_tensors="pt"
            )
            image_inputs = image_inputs.to(self.device)
            image_inputs["pixel_values"] = image_inputs["pixel_values"].half()
            inputs = {**image_inputs, **text_inputs}
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = self.model(**inputs)
            # target_sizes expects (h, w) tuples for each image in batch
            target_sizes = [img.size[::-1] for img in batch_imgs]
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids=text_inputs["input_ids"],
                threshold=self.conf.threshold,
                target_sizes=target_sizes,
            )
            # drop results for dummy images
            if num_additional_dummy_images > 0:
                results = results[:-num_additional_dummy_images]
            # results: list of dict (one per image)
            for res in results:
                detections = [
                    Detection(
                        bbox=tuple(box.tolist()), confidence=score.item(), label=label
                    )
                    for box, score, label in zip(
                        res["boxes"],
                        res["scores"],
                        res["text_labels"],
                    )
                ]
                detections_list.append(detections)
            idx += batch_size
        return detections_list

    def warmup(self, iterations: int = 3) -> None:
        logger.info("GroundingDINO warming up with %d iterations", iterations)
        start_time = time.time()
        self.warming_up = True
        try:
            dummy_image = Image.new("RGB", (512, 512), color="white")
            for _ in range(iterations):
                self.detect([dummy_image], "object .")
        finally:
            self.warming_up = False
            self.warmed_up = True
        logger.info("GroundingDINO warmed up in %.2fs", time.time() - start_time)
