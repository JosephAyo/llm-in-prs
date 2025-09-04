# Copilot Agent Prompt

You are assisting with a research project on textual analysis of LLM-generated descriptions.

## Task
1. Navigate to the `textual-features/` folder.
2. Check all Python scripts in this folder:
   - Verify imports are correct (`scikit-learn`, `numpy`, etc.).
   - Ensure functions have docstrings explaining purpose, inputs, and outputs.
   - Look for duplicate or redundant code and suggest refactoring.
   - Check that each script can be executed without errors in Python 3.12.
3. Run an analysis by:
   - Importing the relevant scripts into the Jupyter notebook inside `analysis/`.
   - Demonstrating example runs of the functions (e.g., cosine similarity).
   - Producing summary results (tables/plots) showing similarities or other textual features.

## Deliverables
- A report (inside the Jupyter notebook in `analysis/`) summarizing:
  - Which scripts were found and what they implement.
  - Any issues (imports, missing docstrings, redundancy).
  - Results of running feature extraction and similarity measures.
- Suggestions for integrating all scripts into a unified pipeline.

## Execution
- Use `pyenv exec` to run all Python scripts in `textual-features/`.
- Ensure results (logs, errors, outputs) are captured in the notebook for reproducibility. checkout log_activity in detection/detection_on_prompt_variations.py
- If dependencies are missing, note them and suggest updates to `requirements.txt`.

## Integration
- Create a summary function (e.g., `run_all_features()`) that aggregates outputs from all scripts.
- Demonstrate this unified pipeline inside the Jupyter notebook.
- Save outputs (similarity scores, tables, plots) into the `analysis/outputs/` folder.


## Notes
- Ask questions when you need clarification
- Later stages may extend to CodeBERT embeddings for conceptual similarity.
- Keep outputs concise but clear: tables, plots, and metrics are preferred.

