# multimodal_utils.py

from typing import List, Optional
from io import BytesIO
from PIL import Image
import tempfile
import os

# Placeholders – you must implement with your chosen open-source models.
# For example:
# - audio_to_text: local Whisper or another STT model
# - image_to_text: local image-LLM (LLaVA, Moondream, etc.)
# - video_to_text: extract frames -> image_to_text on each frame
# - pdf_to_text: PyPDF2 / pdfplumber; txt/md just decode as text


def audio_to_text(audio_bytes: bytes, filetype: str) -> str:
    """
    TODO: implement with your local STT model.
    For now, just return a placeholder.
    """
    return "[Audio transcription not implemented yet]"


def image_to_text(image: Image.Image) -> str:
    """
    TODO: send image to your local vision model and get description.
    """
    return "[Image description not implemented yet]"


def video_to_text(video_bytes: bytes, filetype: str) -> str:
    """
    TODO: extract frames and run image_to_text on a subset.
    """
    return "[Video analysis not implemented yet]"


def pdf_to_text(pdf_bytes: bytes) -> str:
    """
    TODO: implement with PyPDF2 or pdfplumber.
    """
    return "[PDF text extraction not implemented yet]"


def txt_to_text(raw_bytes: bytes) -> str:
    return raw_bytes.decode("utf-8", errors="ignore")


def build_multimodal_context(
    audio_file,
    image_files,
    video_file,
    other_files,
):
    """
    Convert all uploaded media into a single text block describing them.
    """
    parts: List[str] = []

    # ---- audio ----
    if audio_file is not None:
        audio_bytes = audio_file.read()
        audio_text = audio_to_text(audio_bytes, audio_file.type)
        parts.append(f"[AUDIO TRANSCRIPT]\n{audio_text}\n")

    # ---- images ----
    if image_files:
        for i, img_file in enumerate(image_files, start=1):
            img = Image.open(img_file)
            desc = image_to_text(img)
            parts.append(f"[IMAGE {i} DESCRIPTION]\n{desc}\n")

    # ---- video ----
    if video_file is not None:
        video_bytes = video_file.read()
        video_text = video_to_text(video_bytes, video_file.type)
        parts.append(f"[VIDEO DESCRIPTION]\n{video_text}\n")

    # ---- other files ----
    if other_files:
        for f in other_files:
            name = f.name
            data = f.read()
            if name.lower().endswith(".pdf"):
                text = pdf_to_text(data)
            else:
                text = txt_to_text(data)
            parts.append(f"[FILE: {name}]\n{text}\n")

    if not parts:
        return ""

    full_context = "\n\n".join(parts)
    return full_context
