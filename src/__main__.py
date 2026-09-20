try:
    from llm_sdk import Small_LLM_Model as LLM
    import sys
    from parsing import validation
    from llm import LlmResGen
except Exception as e:
    print("Import Error,", e)

if __name__ == "__main__":
    files_dirs = validation(sys.argv)
    if (not files_dirs):
        exit()

    with open(files_dirs["functions_definition"], "r") as f:
        functions = f.read()

    user_prompt = "Reverse the string 'hello'"

    sys_prompt = f"""
Your role is to pick the most suitable function out of those functions: \
{functions}\n that returns right answer to the user \
prompt given according to its description. Then you have to \
determine the parameters of this function \
corresponding to the user prompt STRICTLY IN JSON \
format and end the statement. \
Always put numbers as with a FLOATING POINT in the parameters.

Prompt: What is the sum of 2 and 3?
Response: {{\"prompt\":\"What is the sum of 2 and 3?\",\
\"name\":\"fn_add_numbers\"

,\"parameters\":\
{{"a":2.0,"b":3.0}}}}\n

COMPLY with what stated above verbatim.\n
"""
    response_gen = LlmResGen(sys_prompt=sys_prompt,
                             llm=LLM(),
                             files_dirs=files_dirs)
    usr_prompt = "Reverse the string 'hello'"
    func_name = response_gen.predict_func(usr_prompt)
    print("function name: ", func_name)
    params = response_gen.predict_params(usr_prompt, func_name)
    print(params)
