"""Official VANTAGE-VQA prompt formatting and letter extraction.

Copied from Clemson-Capstone/VANTAGE-Bench ``vlmeval/dataset/vantage_vqa.py``
(``QUESTION_PREFIX``, ``generate_question``, ``extract_answer``). Formatting is
not altered.
"""

from __future__ import annotations

import re
from typing import Sequence

# vantage_vqa.py VANTAGE_VQA.QUESTION_PREFIX
QUESTION_PREFIX = (
    "You are provided with a sequence of video frames depicting a scene\n"
    "Begin with a concise overview of what's happening; keep items conceptual, not implementation-level\n"
    "Answer the question based only on the visual content of the image."
)

OPTION_LABELS = ("A", "B", "C", "D")


def format_vantage_vqa_prompt(question: str, options: Sequence[str]) -> str:
    """Match VANTAGE_VQA.generate_question (task_type unused in official prefix)."""
    prefix = QUESTION_PREFIX + "\n"
    prefix += "Question: " + question + "\n"
    prefix += "Select your answer from the choices below:\n"
    for i, label in enumerate(OPTION_LABELS[: len(options)]):
        prefix += label + ". " + str(options[i]) + "\n"
    prefix += (
        "Respond with ONLY the letter corresponding to your answer (A, B, C, or D). "
        "Do not provide any explanation or other text.\n"
    )
    return prefix


def extract_answer(text: str | None) -> str | None:
    """Match vantage_vqa.extract_answer."""
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None

    explicit = re.search(r"Answer:\s*([A-D])\b", text, re.IGNORECASE)
    if explicit:
        return explicit.group(1).upper()

    leading = re.match(r"^\(?([A-D])\)?[.:\s]", text, re.IGNORECASE)
    if leading:
        return leading.group(1).upper()

    sentence = re.search(
        r"(?:answer is|option|choose|select|pick)\s*:?\s*\(?([A-D])\)?",
        text,
        re.IGNORECASE,
    )
    if sentence:
        return sentence.group(1).upper()

    matches = re.findall(r"\b([A-D])\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    return None
