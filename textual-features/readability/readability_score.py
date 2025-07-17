import subprocess
import os
import sys


def get_readability_score(text):
    # Find absolute path to the readability.jar
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jar_path = os.path.join(script_dir, "readability.jar")

    # Construct classpath properly
    cp = f"{jar_path}"  # Java will treat this as the full JAR path

    result = subprocess.run(
        ["java", "-cp", cp, "ca.usask.cs.text.readability.FleschKincaidReadingEase", text],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise Exception(f"Java error: {result.stderr.strip()}")
    return float(result.stdout.strip())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python readability_score.py \"Your text here...\"")
        sys.exit(1)

    text = " ".join(sys.argv[1:])
    print(f"{get_readability_score(text):.4f}")
