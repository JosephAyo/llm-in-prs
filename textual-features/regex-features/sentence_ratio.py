import sys
import re

SENTENCE_END_REGEX = re.compile(r"[?!.]($|\s)")

def get_sentence_ratio(text):
    sentence_matches = len(SENTENCE_END_REGEX.findall(text))
    total_tokens = len(text.split())
    if total_tokens == 0:
        return 0.0
    return sentence_matches / total_tokens

def main():
    if len(sys.argv) < 2:
        print("Usage: python sentence_ratio.py \"Your text here...\"")
        sys.exit(1)

    text = " ".join(sys.argv[1:])
    ratio = get_sentence_ratio(text)
    print(f"{ratio:.4f}")

if __name__ == "__main__":
    main()
