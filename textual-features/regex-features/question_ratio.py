import sys
import re

QUESTION_REGEX = re.compile(r"[?]($|\s)")

def get_question_ratio(text):
    # Count how many question marks appear followed by space or end of string
    question_matches = len(QUESTION_REGEX.findall(text))
    # Count total tokens (split by whitespace)
    total_tokens = len(text.split())
    if total_tokens == 0:
        return 0.0
    return question_matches / total_tokens

def main():
    if len(sys.argv) < 2:
        print("Usage: python question_ratio.py \"Your text here...\"")
        sys.exit(1)

    text = " ".join(sys.argv[1:])
    ratio = get_question_ratio(text)
    print(f"{ratio:.4f}")

if __name__ == "__main__":
    main()
