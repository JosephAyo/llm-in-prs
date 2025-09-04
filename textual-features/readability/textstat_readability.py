import re
import textstat


def clean_text(text: str) -> str:
    """
    Clean up text before readability analysis.

    - Removes inline code blocks wrapped in backticks.
    - Removes code-like tokens such as function calls, snake_case, or CamelCase.
    - Collapses multiple spaces into one.

    Args:
        text (str): Raw text, possibly containing code artifacts.

    Returns:
        str: Cleaned natural language text.
    """
    # Remove inline code blocks or backticks
    text = re.sub(r"`[^`]*`", "", text)
    # Remove code-like tokens (function(), variables)
    text = re.sub(r"[A-Za-z_]+\([^)]*\)", "", text)
    text = re.sub(r"[_A-Za-z0-9]+", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_readability_features(text: str) -> dict:
    """
    Compute readability metrics for a given text using textstat.

    Args:
        text (str): Raw text string.

    Returns:
        dict: Dictionary of readability scores.
    """
    cleaned = clean_text(text)
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(cleaned), # type: ignore
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(cleaned),# type: ignore
        "gunning_fog": textstat.gunning_fog(cleaned),# type: ignore
        "automated_readability_index": textstat.automated_readability_index(cleaned),# type: ignore
    }


if __name__ == "__main__":
    description = "This function should validate user input before processing."
    features = compute_readability_features(description)
    print("Readability features:")
    for metric, score in features.items():
        print(f"{metric}: {score}")
