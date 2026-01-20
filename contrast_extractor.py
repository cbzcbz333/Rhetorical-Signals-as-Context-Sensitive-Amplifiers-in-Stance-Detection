import re
import nltk
from nltk.tokenize import sent_tokenize

# 如果本地没装
# nltk.download('punkt')

CONTRAST_MARKERS = [
    r"\bbut\b",
    r"\bhowever\b",
    r"\byet\b",
    r"\balthough\b",
    r"\bthough\b",
    r"\bwhereas\b",
    r"\bwhile\b",
    r"\bon the one hand\b",
    r"\bon the other hand\b",
    r"\bnot\b.*\bbut\b",
    r"\brather than\b"
]

CONTRAST_REGEX = re.compile("|".join(CONTRAST_MARKERS), re.IGNORECASE)


def extract_contrast_features(text: str) -> dict:
    """
    提取对立 / 对比修辞特征
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "contrast_marker_count": 0,
            "contrast_sentence_ratio": 0.0,
            "has_contrast_structure": 0
        }

    sentences = sent_tokenize(text)
    if not sentences:
        return {
            "contrast_marker_count": 0,
            "contrast_sentence_ratio": 0.0,
            "has_contrast_structure": 0
        }

    marker_count = 0
    contrast_sentences = 0

    for sent in sentences:
        matches = CONTRAST_REGEX.findall(sent)
        if matches:
            marker_count += len(matches)
            contrast_sentences += 1

    return {
        "contrast_marker_count": marker_count,
        "contrast_sentence_ratio": contrast_sentences / len(sentences),
        "has_contrast_structure": int(contrast_sentences > 0)
    }
