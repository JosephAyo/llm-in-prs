from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -------------------------------
# Preprocessing & Similarity Metric
# -------------------------------
def compute_description_code_similarity(description, changed_lines):
    """
    Compute conceptual similarity between a description (e.g., LLM generated)
    and a list of changed source code lines.

    Args:
        description (str): The generated description text.
        changed_lines (list[str]): List of code lines (added, deleted, modified).

    Returns:
        float: Maximum cosine similarity between description and changed lines.
    """

    # Entities = description + code lines
    documents = [description] + changed_lines

    # TF-IDF vectorization (basic tokenization & stopword removal)
    vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'\b\w+\b')
    tfidf_matrix = vectorizer.fit_transform(documents)

    desc_vec = tfidf_matrix[0]  # type: ignore
    code_vecs = tfidf_matrix[1:]  # type: ignore


    # Compute cosine similarity
    sims = cosine_similarity(desc_vec, code_vecs)[0]


    # Compute cosine similarity
    sims = cosine_similarity(desc_vec, code_vecs)[0]

    # Return maximum similarity across all changed lines
    return float(np.max(sims)) if len(sims) > 0 else 0.0


# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    description = "This code should handle empty data properly."
    changed_lines = [
        "def process_data(data):",
        "    if not data: return None",
        "    result = data.strip().lower()",
        "    return result",
    ]


    score = compute_description_code_similarity(description, changed_lines)
    print("Max cosine similarity:", score)