try:
    from llm_sdk import Small_LLM_Model as LLM
    import sys
    from parsing import validation
    from llm import LlmResGen
    import json
except Exception as e:
    print("Import Error,", e)

if __name__ == "__main__":
    files_dirs = validation(sys.argv)
    if (not files_dirs):
        exit()
    with open(files_dirs["input"], "r") as f:
        prompts = json.loads(f.read())
    response_gen = LlmResGen(llm=LLM(),
                             files_dirs=files_dirs)
    answer = []
    for prompt in prompts:
        answer.append(response_gen.gen_answer_obj(prompt["prompt"]))
    answer_json = json.dumps(answer, indent=4)
    print(answer_json)
