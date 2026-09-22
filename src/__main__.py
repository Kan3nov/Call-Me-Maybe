try:
    from llm_sdk import Small_LLM_Model as LLM
    from pathlib import Path
    import sys
    from parsing import validation
    from llm import LlmResGen
    import json
except Exception as e:
    print("Import Error,", e)
    exit()

if __name__ == "__main__":
    files_dirs = validation(sys.argv)
    if (not files_dirs):
        exit()
    try:
        with open(files_dirs["input"], "r") as f:
            prompts = json.loads(f.read())
    except Exception as e:
        print("Error reading user prompts", e)

    response_gen = LlmResGen(llm=LLM(),
                             files_dirs=files_dirs)
    answer = []
    for prompt in prompts:
        answer.append(response_gen.gen_answer_obj(prompt["prompt"]))
        print(answer[-1])
    answer_json = json.dumps(answer, indent=4)
    file = Path(files_dirs["output"])
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(files_dirs["output"], "w") as f:
        f.write(answer_json)
