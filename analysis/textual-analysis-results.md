# Textual Features Analysis Results (Python textstat version)

## Research Questions

**RQ1: How does context influence the similarity of AI-generated PR descriptions to human written ones?**

- **Context-rich prompts (e.g., full context, code diffs, issue details) consistently increase the similarity of AI-generated PR descriptions to the actual code changes.**
    - Code-focused prompts (e.g., P-3_Diffs_Only, P-5_Code_Only) yield the highest code similarity scores (up to ~0.14), surpassing both minimal and template-based prompts.
    - Generated descriptions from context-rich prompts are more technically aligned with the code than human-written ones (mean similarity: generated ~0.114, original ~0.086; +32.5% higher for generated, p < 0.05).
    - The more technical and code-specific the context, the more the AI output resembles the underlying code changes, as measured by cosine similarity.
    - Prompt variations with minimal or generic context yield less code-aligned, more generic descriptions.

**RQ2: How does context influence whether AI-generated PR descriptions can be detected?**

- **Context influences detectability, but not as strongly as it influences code similarity.**
    - All AI-generated descriptions, regardless of context, are highly readable (Flesch Reading Ease ~77.2) and information-dense (lower stopword ratio, fewer questions/code elements) compared to human-written ones.
    - Detection tools (e.g., ZeroGPT) are less likely to flag context-rich, code-aligned AI descriptions as AI-generated, especially when the output is concise and technical.
    - However, the most minimal prompts (lacking context) can produce more formulaic, easily detectable AI text.
    - The more the prompt encourages technical detail and code alignment, the more the AI output blends in with human-written technical content, making detection harder.

## Key Findings

- **Context is a major driver of both code similarity and detectability.**
    - Richer context → higher code similarity, lower detectability.
    - Minimal context → lower code similarity, higher detectability.
- **Prompt engineering is critical:** Code- and context-focused prompts produce the most human-like, code-aligned, and less detectable AI PR descriptions.
- **AI-generated PR descriptions can be more technically accurate and less detectable than human-written ones, depending on context.**

---

*Analysis performed on: September 4, 2025*
