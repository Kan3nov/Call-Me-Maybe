*This activity has been created as part of the 42 curriculum by klafi*

# Description
Large Language Models (LLMs) are powerful at understanding and generating human language, but they don’t naturally produce structured, machine-executable output. Function calling bridges this gap by translating natural language requests into precise function calls with typed arguments. The function calling system doesn’t answer the question directly. Instead, it provides the toolsto solve it: the right function name and the correct arguments with proper types.

Small language models (like the 0.6B parameter model I am using) are notoriously unreliable at generating structured output. When prompted to produce JSON, they might succeed only 30% of the time. Yet production systems achieve 99%+ reliability with these same small models.

How? The answer lies in constrained decoding — a technique that guides the model’s output token-by-token to guarantee valid structure, without relying on prompting alone.

In this project, I use constrained decoding techniques to ensure the model implements function calling properly such that all functions calls for each propmt are serialized in JSON format.

# Instructions
In this project, the uv package manager was used. uv is characterized by its speed due to its Rust implementation which makes it a very wise choice.

Those are all the Makefile rules you would need to install dependancies and run the project:
`make install`: Installs all the needed packages for the program to run.
`make run`: Executes the program with default input files' locations.
`make debug`: Executes the program in debug mode.
`make clean`: Deletes all temporary files for cleaning the  environment.
`make all`(default): runs the program and installs all needed dependancies.

By default, the program will read input files from the data/input/ directory and write output to the data/output/ directory. You can optionally specify custom paths using the --input and --output arguments.

To run the program with your own arguments:
```
uv run python -m src [--functions_definition
<function_definition_file>] [--input <input_file>] [--output
<output_file>]
```
# Algorithm explanation
For constrained decoding, my program build the JSON structure single-handedly to ensure 100% JSON-compatible fomrat. The model's only role is filling the function name and parameter values in the structure. To further assure the output validity, I used the masking technique to prevent the model from predicting invalid tokens for each parameter type by setting their logit value to negative infinity. For example, for integer parameters, all tokens not including digits, a comma, or a closing curly brace are immediately eleminated.
# Desgin decisions
- **Greedy, masked arg-max instead of sampling**: since the goal is 100% valid, deterministic structured output rather than creative text, the token selection is a masked arg-max at every step rather than temperature-based sampling.
- **Whitelist-first vocabulary control**: rather than trying to fix up invalid JSON after generation (regex repair, retries, etc.), invalid tokens are prevented from ever being chosen. This is what gives the 100%-valid-JSON guarantee instead of a "usually valid" guarantee.
- **Function-scoped prompt context**: when generating function names, the system prompt is narrowed to show only the tokens including string subsets of all function names, which reduces confusion from irrelevant tokens without requiring a per-field hint system.
- **fn_undefined fallback**: the parser automatically injects a synthetic fn_undefined entry into the function dictionary so the model always has a safe fallback when no real function matches the prompt.
# Performance analysis
The program is able to generate all answers in less than 5 minutes with more than 90% accuracy which makes the program rather reliable.
# Challenges faced
- For regex-dependent prompts, the model used to hallucinate and give obscure answers. So I gave examples to instruct the model.
- The unability of recognizing tokens by their id via the vocab file. Instead, I used the decode predefined function from the llm_sdk. 
# Testing Startegy
Ran a variety of prompts and covered all conceivable edge cases like emtry prompts, large numbers, and ambiguous prompts.
# Example usage
```
uv run python -m src
--functions_definition data/input/functions_definition.json
--input data/input/function_calling_tests.json
--output data/output/function_calls.json
```
# Resouces
- **UV Package Manager:**
    - https://www.datacamp.com/tutorial/python-uv
    - https://www.reddit.com/r/learnpython/comments/1kkw9ou/installing_dependencies_from_a_project_using_uv/
- **Word Embeddings**
    -  https://youtu.be/wgfSDrqYMJ4?si=mhKfx7N3Mk3R2fLn
- **Transformers**
    - https://youtu.be/ZXiruGOCn9s?si=PK0-TXqML7y810Y-
- **User Prompts Vs. System Prompts**
    - https://blog.promptlayer.com/system-prompt-vs-user-prompt-a-comprehensive-guide-for-ai-prompts/
- **Masking**
    - https://jykroo.medium.com/mastering-masking-in-machine-learning-the-backbone-of-transformer-models-af3cbc48b9f2
- **AI Usage**
    - Was used to help with designing system prompts.
# Final message
We extend our deepest thanks and appreciation to the esteemed Mr. Mohammad Adel Khashan and Mr. Rashed Alrawajbeh —may God marry him off and watch over him—for his pivotal assistance in bringing this project to fruition. Happy birthday to you.