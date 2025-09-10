import re
import textstat


def clean_text(text: str) -> str:
    """
    Clean up text before readability analysis.

    - Removes inline code blocks wrapped in backticks.
    - Removes code-like tokens such as function calls, snake_case, or CamelCase.
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
    
    # Remove function calls with parentheses
    text = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\([^)]*\)", " ", text)
    
    # Remove URLs and file paths
    text = re.sub(r"https?://[^\s]+", " ", text)
    text = re.sub(r"[/\\][A-Za-z0-9_./\\-]+", " ", text)
    
    # Remove specific code-like patterns (but preserve normal words)
    text = re.sub(r"\b[A-Z]+_[A-Z_]+\b", " ", text)        # CONSTANT_NAMES
    text = re.sub(r"\b[a-z]+_[a-z_]+\b", " ", text)        # snake_case variables
    text = re.sub(r"\b[A-Z][a-z]*[A-Z][A-Za-z]*\b", " ", text)  # CamelCase
    text = re.sub(r"\b[A-Za-z0-9]+\.[A-Za-z0-9]+\b", " ", text)  # object.method or file.ext
    
    # Remove standalone numbers and version strings
    text = re.sub(r"\b\d+(\.\d+)*\b", " ", text)
    
    # Remove special characters that are not punctuation
    text = re.sub(r"[{}[\]|\\<>@#$%^&*+=~]", " ", text)
    
    # Collapse multiple spaces and clean up
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
