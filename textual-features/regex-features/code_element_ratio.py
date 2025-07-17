import sys
import re

# Matches:
# - triple backticks with any characters in between (including newlines)
# - inline backticks with one or more non-whitespace chars
# - inline backticks with multiple whitespace/non-whitespace chars (though the original is a bit ambiguous)
CODE_REGEX = re.compile(r"(`{3}[\s\S]+?`{3}|`[^`\s]+`|`[\s\S]+?`)")

def get_code_element_ratio(text):
    code_matches = len(CODE_REGEX.findall(text))
    total_tokens = len(text.split())
    if total_tokens == 0:
        return 0.0
    return code_matches / total_tokens

def main():
    if len(sys.argv) < 2:
        print("Usage: python code_element_ratio.py \"Your text here...\"")
        sys.exit(1)

    text = " ".join(sys.argv[1:])
    ratio = get_code_element_ratio(text)
    print(f"{ratio:.4f}")

if __name__ == "__main__":
    main()
