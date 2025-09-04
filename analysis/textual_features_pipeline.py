"""
Unified Textual Features Pipeline

This module provides a comprehensive pipeline for extracting textual features
from LLM-generated descriptions, integrating all scripts from textual-features/.

Usage:
    from textual_features_pipeline import TextualFeatureExtractor
    
    extractor = TextualFeatureExtractor()
    results = extractor.run_all_features(texts)
"""

import sys
import os
import subprocess
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextualFeatureExtractor:
    """
    Unified pipeline for extracting textual features from text descriptions.
    
    This class integrates all the individual feature extraction scripts:
    - Stop word ratio calculation
    - Regex-based features (code elements, questions, sentences)
    - Readability scores (via Java dependency)
    - Cosine similarity with code samples
    """
    
    def __init__(self, textual_features_path=None):
        """
        Initialize the TextualFeatureExtractor.
        
        Args:
            textual_features_path (str): Path to textual-features directory
        """
        if textual_features_path is None:
            # Try to find textual-features directory relative to this script
            current_dir = Path(__file__).parent
            self.textual_features_path = current_dir.parent / 'textual-features'
        else:
            self.textual_features_path = Path(textual_features_path)
        
        self.stop_words = None
        self._load_dependencies()
    
    def _load_dependencies(self):
        """Load all required dependencies and scripts."""
        # Load stop words
        stopword_file = self.textual_features_path / 'stop-word-ratio' / 'stop-words-english-total.txt'
        if stopword_file.exists():
            with open(stopword_file, 'r', encoding='utf-8') as f:
                self.stop_words = set(word.strip().lower() for word in f if word.strip())
    
    def get_stop_word_ratio(self, text):
        """Calculate stop word ratio for given text."""
        if not self.stop_words:
            return 0.0
        
        words = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
        if not words:
            return 0.0
        stop_word_count = sum(1 for word in words if word in self.stop_words)
        return stop_word_count / len(words)
    
    def get_code_element_ratio(self, text):
        """Calculate code element ratio using regex."""
        CODE_REGEX = re.compile(r"(`{3}[\s\S]+?`{3}|`[^`\s]+`|`[\s\S]+?`)")
        code_matches = len(CODE_REGEX.findall(text))
        total_tokens = len(text.split())
        return code_matches / total_tokens if total_tokens > 0 else 0.0
    
    def get_question_ratio(self, text):
        """Calculate question ratio using regex."""
        QUESTION_REGEX = re.compile(r"[?]($|\s)")
        question_matches = len(QUESTION_REGEX.findall(text))
        total_tokens = len(text.split())
        return question_matches / total_tokens if total_tokens > 0 else 0.0
    
    def get_sentence_ratio(self, text):
        """Calculate sentence ending ratio using regex."""
        SENTENCE_END_REGEX = re.compile(r"[?!.]($|\s)")
        sentence_matches = len(SENTENCE_END_REGEX.findall(text))
        total_tokens = len(text.split())
        return sentence_matches / total_tokens if total_tokens > 0 else 0.0
    
    def get_readability_score(self, text):
        """Get readability score using textstat_readability (flesch_reading_ease)."""
        try:
            import importlib.util
            import sys
            import pathlib
            tf_path = pathlib.Path(__file__).parent.parent / 'textual-features' / 'readability' / 'textstat_readability.py'
            spec = importlib.util.spec_from_file_location("textstat_readability", str(tf_path))
            if spec is None or spec.loader is None:
                return None
            tf_module = importlib.util.module_from_spec(spec)
            sys.modules["textstat_readability"] = tf_module
            spec.loader.exec_module(tf_module)
            features = tf_module.compute_readability_features(text)
            return features.get("flesch_reading_ease", None)
        except Exception:
            return None
    
    def compute_cosine_similarity(self, description, changed_lines):
        """
        Compute cosine similarity between description and code lines.
        
        Args:
            description (str): The description text
            changed_lines (list): List of code lines
            
        Returns:
            float: Maximum cosine similarity score
        """
        if not changed_lines:
            return 0.0
        
        try:
            documents = [description] + changed_lines
            vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'\b\w+\b')
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            desc_vec = tfidf_matrix[0]  # type: ignore
            code_vecs = tfidf_matrix[1:]  # type: ignore
            
            sims = cosine_similarity(desc_vec, code_vecs)[0]
            return float(np.max(sims)) if len(sims) > 0 else 0.0
        except Exception:
            return 0.0
    
    def extract_features(self, text):
        """
        Extract all textual features for a single text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Dictionary containing all extracted features
        """
        features = {
            'text': text,
            'length': len(text),
            'word_count': len(text.split()) if text.strip() else 0
        }
        
        # Basic statistics
        features['sentence_count'] = len([s for s in text.split('.') if s.strip()])
        features['avg_word_length'] = (
            np.mean([len(word) for word in text.split()]) 
            if text.split() else 0
        )
        
        # Textual features
        features['stopword_ratio'] = self.get_stop_word_ratio(text)
        features['code_element_ratio'] = self.get_code_element_ratio(text)
        features['question_ratio'] = self.get_question_ratio(text)
        features['sentence_ratio'] = self.get_sentence_ratio(text)
        features['readability_score'] = self.get_readability_score(text)
        
        return features
    
    def run_all_features(self, texts, code_samples=None, include_similarity=True):
        """
        Unified pipeline to extract all textual features for a list of texts.
        
        Args:
            texts (list): List of text strings to analyze
            code_samples (dict): Dictionary of code samples for similarity analysis
            include_similarity (bool): Whether to include cosine similarity analysis
        
        Returns:
            pd.DataFrame: DataFrame containing all extracted features
        """
        results = []
        
        for i, text in enumerate(texts):
            result = self.extract_features(text)
            result['text_id'] = i
            
            # Add similarity features if requested
            if include_similarity and code_samples:
                for code_type, code_lines in code_samples.items():
                    similarity = self.compute_cosine_similarity(text, code_lines)
                    result[f'{code_type.lower().replace(" ", "_")}_similarity'] = similarity
            
            results.append(result)
        
        return pd.DataFrame(results)


# Example usage and default code samples
DEFAULT_CODE_SAMPLES = {
    "Bug fix": [
        "if (user == null) throw new NullPointerException();",
        "session.invalidate();",
        'logger.error("Null user in session");'
    ],
    "API enhancement": [
        "@RateLimit(requests = 100, per = TimeUnit.MINUTES)",
        "public ResponseEntity<Object> handleApiCall() {",
        "    return ResponseEntity.ok().build();",
        "}"
    ],
    "Database optimization": [
        "EXPLAIN ANALYZE SELECT * FROM users;",
        "CREATE INDEX idx_user_email ON users(email);",
        "SELECT /*+ USE_INDEX(users, idx_user_email) */ * FROM users;"
    ]
}


def main():
    """Example usage of the TextualFeatureExtractor."""
    # Sample texts
    sample_texts = [
        "Fix null pointer exception in user session manager module.",
        "How should we handle rate limiting for API calls?", 
        "Add SQL query optimization using `EXPLAIN ANALYZE` to identify bottlenecks."
    ]
    
    # Initialize extractor
    extractor = TextualFeatureExtractor()
    
    # Run analysis
    results = extractor.run_all_features(
        sample_texts, 
        code_samples=DEFAULT_CODE_SAMPLES,
        include_similarity=True
    )
    
    print("Textual Features Analysis Results:")
    print("=" * 50)
    print(results.round(4))
    
    return results


if __name__ == "__main__":
    main()
