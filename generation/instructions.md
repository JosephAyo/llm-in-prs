## Requirements
The generation/prompting-guide.md doc is a excerpt from https://cookbook.openai.com/examples/gpt4-1_prompting_guide, I want to update my prompts in generation/generate_with_prompt_variations.py. The plan is to use some and not all the suggestions from the doc.

## Additional information
### Tips that should be considered
1. Prompt Structure.
2. Delimiters
3. Prompt Organization, "ideally place your instructions at both the beginning and end of the provided context, as we found this to perform better than only above or below" for long context prompts.
4. Instruction following structure, outline how the model should carry out the task (try to follow the way a human would go about writing description)


### Tips that can be ignored.
1. Tool-calling


## Instructions
- First check the source link https://cookbook.openai.com/examples/gpt4-1_prompting_guide, study and understand the prompt guide.
- Tell me your strategy to update the prompts in the code and ask for confirmation.
- Update the prompts. Achieving this task may require constructing the prompts separately for each prompt-level (i.e. P-*), minimal P-1 for example may not needs instructions at both end.
- Please ask questions whenever you need clarification


## Post-completion task.
don't delete instructions.md
write a summary.md and it should contain a write commit message that follows conventional commits.

## relevant commands to run scripts

always use pyenv exec to run scripts, e.g.:
```
pyenv exec python script.py
`