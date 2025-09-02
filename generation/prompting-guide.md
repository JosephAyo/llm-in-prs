
## Agentic Workflows
Agentic workflows don't necessarily apply to the task.
### System Prompt Reminders
This is recommended in order to utilize the agentic capabilities of GPT-4.1, but it may not be need for the intended task.
1. Persistence: this ensures the model understands it is entering a multi-message turn, and prevents it from prematurely yielding control back to the user.
2. Tool-calling: this encourages the model to make full use of its tools, and reduces its likelihood of hallucinating or guessing an answer. <b>This may be useful</b>
3. Planning [optional]: if desired, this ensures the model explicitly plans and reflects upon each tool call in text, instead of completing the task by chaining together a series of only tool calls

### Tool Calls
The intended task does not require a tool call.

## Long context
### Prompt Organization
Especially in long context usage, placement of instructions and context can impact performance. If you have long context in your prompt, ideally place your instructions at both the beginning and end of the provided context, as we found this to perform better than only above or below. If you’d prefer to only have your instructions once, then above the provided context works better than below. <i>[this should be considered, especially for prompts that have long contexts]</i>

## Chain of Thought
As mentioned above, GPT-4.1 is not a reasoning model, but prompting the model to think step by step (called “chain of thought”) can be an effective way for a model to break down problems into more manageable pieces, solve them, and improve overall output quality, with the tradeoff of higher cost and latency associated with using more output tokens. Example <i>[should be considered, especially for more complex prompt levels (P-*)]</i>:
```
# Reasoning Strategy
1. Query Analysis: Break down and analyze the query until you're confident about what it might be asking. Consider the provided context to help clarify any ambiguous or confusing information.
2. Context Analysis: Carefully select and analyze a large set of potentially relevant documents. Optimize for recall - it's okay if some are irrelevant, but the correct documents must be in this list, otherwise your final answer will be wrong. Analysis steps for each:
	a. Analysis: An analysis of how it may or may not be relevant to answering the query.
	b. Relevance rating: [high, medium, low, none]
3. Synthesis: summarize which documents are most relevant and why, including all documents with a relevance rating of medium or higher.

# User Question
{user_question}

# External Context
{external_context}

First, think carefully step by step about what documents are needed to answer the query, closely adhering to the provided Reasoning Strategy. Then, print out the TITLE and ID of each document. Then, format the IDs into a list.
```

## [Instruction Following](https://cookbook.openai.com/examples/gpt4-1_prompting_guide#4-instruction-following) 
GPT-4.1 exhibits outstanding instruction-following performance, which developers can leverage to precisely shape and control the outputs for their particular use cases. Example
```
SYS_PROMPT_CUSTOMER_SERVICE = """You are a helpful customer service agent working for NewTelco, helping a user efficiently fulfill their request while adhering closely to provided guidelines.

# Instructions
- Always greet the user with "Hi, you've reached NewTelco, how can I help you?"
- Always call a tool before answering factual questions about the company, its offerings or products, or a user's account. Only use retrieved context and never rely on your own knowledge for any of these questions.
    - However, if you don't have enough information to properly call the tool, ask the user for the information you need.
- Escalate to a human if the user requests.
- Do not discuss prohibited topics (politics, religion, controversial current events, medical, legal, or financial advice, personal conversations, internal company operations, or criticism of any people or company).
- Rely on sample phrases whenever appropriate, but never repeat a sample phrase in the same conversation. Feel free to vary the sample phrases to avoid sounding repetitive and make it more appropriate for the user.
- Always follow the provided output format for new messages, including citations for any factual statements from retrieved policy documents.
- If you're going to call a tool, always message the user with an appropriate message before and after calling the tool.
- Maintain a professional and concise tone in all responses, and use emojis between sentences.
- If you've resolved the user's request, ask if there's anything else you can help with

# Precise Response Steps (for each response)
1. If necessary, call tools to fulfill the user's desired action. Always message the user before and after calling a tool to keep them in the loop.
2. In your response to the user
    a. Use active listening and echo back what you heard the user ask for.
    b. Respond appropriately given the above guidelines.

# Sample Phrases
## Deflecting a Prohibited Topic
- "I'm sorry, but I'm unable to discuss that topic. Is there something else I can help you with?"
- "That's not something I'm able to provide information on, but I'm happy to help with any other questions you may have."

## Before calling a tool
- "To help you with that, I'll just need to verify your information."
- "Let me check that for you—one moment, please."
- "I'll retrieve the latest details for you now."

## After calling a tool
- "Okay, here's what I found: [response]"
- "So here's what I found: [response]"

# Output Format
- Always include your final response to the user.
- When providing factual information from retrieved context, always include citations immediately after the relevant statement(s). Use the following citation format:
    - For a single source: [NAME](ID)
    - For multiple sources: [NAME](ID), [NAME](ID)
- Only provide information about this company, its policies, its products, or the customer's account, and only if it is based on information provided in context. Do not answer questions outside this scope.
```

## General Advice
### Prompt Structure
For reference, here is a good starting point for structuring your prompts.
```
# Role and Objective

# Instructions

## Sub-categories for more detailed instructions

# Reasoning Steps

# Output Format

# Examples
## Example 1

# Context

# Final instructions and prompt to think step by step
```

### Delimiters
Markdown: We recommend starting here, and using markdown titles for major sections and subsections (including deeper hierarchy, to H4+). Use inline backticks or backtick blocks to precisely wrap code, and standard numbered or bulleted lists as needed. <i>[use this for sections]</i>