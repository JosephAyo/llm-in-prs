import subprocess
pr_descriptions = [
    # Typical bugfix and maintenance
    "Fix null pointer exception in user session manager module.",
    "Update third-party dependencies and remove deprecated APIs.",
    "Improve error handling for file upload failures on mobile.",
    "Refactor authentication middleware for better readability and reuse.",
    "Add logging and metrics to track database connection pool usage.",
    "Implement pagination for user activity feed endpoint.",
    "Update CI/CD pipeline to include integration tests for payments module.",
    "Remove unused variables and redundant code in analytics service.",
    "Fix layout issues on iOS Safari for checkout page.",
    "Add feature flag for experimental search ranking algorithm?",  # question

    # Short, simple statements
    "Optimize CSS for faster rendering.",
    "Fix typo in README.",

    # Questions
    "How should we handle rate limiting for API calls?",
    "Is it possible to support multi-factor authentication in this module?",
    "What impact will this change have on legacy systems?",

    # Long descriptive sentences
    "Refactor the entire notification system to support asynchronous message delivery and handle edge cases such as network failures or user unavailability more gracefully.",
    "Improve database query performance by introducing indexing on frequently accessed columns and reducing redundant joins in complex report generation scripts.",

    # Code elements (inline and blocks)
    "Add SQL query optimization using `EXPLAIN ANALYZE` to identify bottlenecks.",
    "Fix parsing bug in JSON serializer:\n```json\n{\n  \"key\": \"value\",\n  \"number\": 123\n}\n```",
    "Update `UserService` class to implement the new interface `IUserManager`.",
    "Refactor code block:\n```python\ndef process_data(data):\n    # process\n    pass\n```",

    # Mixed punctuation and questions
    "Can we improve the caching mechanism? It seems to expire too soon!",
    "Why does the app crash when loading large files?",
    "Improve logging for debug purposes. Are we capturing enough context?",

    # Technical jargon
    "Implement OAuth2 authorization flow with refresh tokens for enhanced security.",
    "Add support for WebSocket connections to enable real-time updates.",

    # Edge cases / empty or nonsense
    "",
    "    ",
    "asdf1234 ??? !!! ```code```",

    # Multiple sentences in one PR description
    "Fix memory leak in image processing module. This was caused by improperly closed streams. Added unit tests.",
    "Remove deprecated endpoints. Clients should migrate to the new API by next release.",
]

def run_python_script(script_path, text):
    result = subprocess.run(
        ["/Users/mac/.pyenv/versions/3.12.9/bin/python", script_path, text],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise Exception(f"Error in {script_path}: {result.stderr.strip()}")
    return float(result.stdout.strip())

def get_stopword_ratio(text):
    return run_python_script("./stop-word-ratio/stopword_ratio.py", text)

def get_readability_score(text):
    return run_python_script("./readability/readability_score.py", text)

def get_question_ratio(text):
    return run_python_script("./regex-features/question_ratio.py", text)

def get_sentence_ratio(text):
    return run_python_script("./regex-features/sentence_ratio.py", text)

def get_code_element_ratio(text):
    return run_python_script("./regex-features/code_element_ratio.py", text)

# Run and print all results
for desc in pr_descriptions:
    try:
        stopword_ratio = get_stopword_ratio(desc)
        readability_score = get_readability_score(desc)
        question_ratio = get_question_ratio(desc)
        sentence_ratio = get_sentence_ratio(desc)
        code_element_ratio = get_code_element_ratio(desc)

        print(f"Description: {desc}")
        print(f"Stop Word Ratio: {stopword_ratio:.4f}")
        print(f"Readability Score: {readability_score:.4f}")
        print(f"Question Ratio: {question_ratio:.4f}")
        print(f"Sentence Ratio: {sentence_ratio:.4f}")
        print(f"Code Element Ratio: {code_element_ratio:.4f}")
        print("-" * 40)

    except Exception as e:
        print(f"Error processing description:\n{desc}\n{e}\n")
