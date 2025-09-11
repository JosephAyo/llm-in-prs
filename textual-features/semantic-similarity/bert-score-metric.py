import re
from bert_score import score


def clean_text(text: str) -> str:
    """
    Clean up text before BERTScore analysis.

    - Removes inline code blocks wrapped in backticks.
    - Removes URLs and file paths.
    - Collapses multiple spaces into one.

    Args:
        text (str): Raw text, possibly containing code artifacts.

    Returns:
        str: Cleaned natural language text.
    """
    # Remove inline code blocks (single and triple backticks)
    text = re.sub(r"`{3}[^`]*`{3}", " ", text)  # Triple backticks (code blocks)
    text = re.sub(r"`[^`]*`", " ", text)        # Single backticks (inline code)

    # Remove URLs and file paths
    text = re.sub(r"https?://[^\s]+", " ", text)
    text = re.sub(r"[/\\][A-Za-z0-9_./\\-]+", " ", text)

    # Collapse multiple spaces and clean up
    text = re.sub(r"\s+", " ", text).strip()

    return text


def compute_bert_score(candidate: str, reference: str, lang: str = "en") -> dict:
    """
    Compute BERTScore metrics for candidate vs reference text.

    Args:
        candidate (str): The generated or test text.
        reference (str): The gold/reference text.
        lang (str): Language code for BERTScore (default: English).

    Returns:
        dict: Dictionary containing precision, recall, and F1 scores.
    """
    cand_clean = clean_text(candidate)
    ref_clean = clean_text(reference)

    # bert-score requires lists of strings
    P, R, F1 = score([cand_clean], [ref_clean], lang=lang)

    return {
        "bert_precision": float(P.mean()),
        "bert_recall": float(R.mean()),
        "bert_f1": float(F1.mean()),
    }


if __name__ == "__main__":
    candidate = "This function validates user input before it is processed."
    reference = "The function should validate user input before processing."

    results = compute_bert_score(candidate, reference)

    print("BERTScore results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
