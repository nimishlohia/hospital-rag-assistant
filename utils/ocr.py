from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_ocr_reader: Any = None


def get_ocr_reader() -> Any:
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            import torch

            use_gpu = torch.cuda.is_available()
            logger.info("Initializing EasyOCR reader (GPU=%s)...", use_gpu)
            _ocr_reader = easyocr.Reader(["en"], gpu=use_gpu)
        except Exception as exc:
            logger.error("Failed to initialize EasyOCR: %s", exc)
            raise RuntimeError(
                f"EasyOCR is not installed or failed to initialize: {exc}. "
                "Run 'pip install easyocr pypdfium2' to enable OCR."
            ) from exc
    return _ocr_reader


def ocr_image(image_input: bytes | Image.Image) -> str:
    """Extract text from an image (bytes or PIL Image) using EasyOCR."""
    reader = get_ocr_reader()
    if isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    else:
        pil_img = image_input.convert("RGB")

    img_np = np.array(pil_img)
    results = reader.readtext(img_np, detail=0)
    return " ".join(results).strip()
