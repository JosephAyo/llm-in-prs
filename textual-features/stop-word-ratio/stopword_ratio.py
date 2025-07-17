import sys
import string
import os


def load_stop_words(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return set(word.strip().lower() for word in f if word.strip())

def get_stop_word_ratio(text, stop_words):
    words = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
    if not words:
        return 0.0
    stop_word_count = sum(1 for word in words if word in stop_words)
    return stop_word_count / len(words)

def main():
    if len(sys.argv) < 2:
        print("Usage: python stopword_ratio.py \"Your text here...\"")
        sys.exit(1)

    text = " ".join(sys.argv[1:])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stopword_file = os.path.join(script_dir, "stop-words-english-total.txt")
    stop_words = load_stop_words(stopword_file)
    ratio = get_stop_word_ratio(text, stop_words)
    print(f"{ratio:.4f}")

if __name__ == "__main__":
    main()
