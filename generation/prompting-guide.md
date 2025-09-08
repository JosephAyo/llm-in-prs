# Task Instructions for VS Code Agent

## Objective
You need to update the **`generation/generate_with_prompt_variations.py`** script used in my research work.  
The goal is to **modify the set of prompt variations** that the script uses, based on the attached image

## Details
1. Locate the script file named `generation/generate_with_prompt_variations.py`
2. Identify the section of the code where **prompt templates / prompt variations** are defined.
3. Update the existing prompt variations, following the structure of the existing implementation.
4. Ensure that:
   - The prompt structure stays the same, the only thing changing is how the components are used
6. Test that the updated prompts are properly integrated and used during generation.

## Deliverables
- Updated `generation/generate_with_prompt_variations.py` script (and any associated config files if applicable).
- Clear inline comments explaining the newly added prompt variations.
- A brief summary in the code comments describing:
  - What was changed,
  - Why it was changed,
  - How to extend it in the future.

## Notes
- Ask questions is anything is vague or unclear
- Keep coding style consistent with the existing script.
- Do not remove or break existing functionality.
- If unsure about the correct place to insert the new prompts, prefer modularity (e.g., add to a configuration structure).


## Post-completion task.
don't delete instructions.md
write a summary.md and it should contain a write commit message that follows conventional commits.

## relevant commands to run scripts

always use pyenv exec to run scripts, e.g.:
```
pyenv exec python script.py
`