try:
    from llm_sdk import Small_LLM_Model as LLM
    from pathlib import Path
    from pydantic import TypeAdapter
    import sys
    from parsing import validation
    from llm import LlmResGen
    from json_schema import Prompt
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
            prompt_adapter = TypeAdapter(list[Prompt])
            content = f.read()
            if (prompt_adapter.validate_json(content)):
                prompts = json.loads(content)
    except Exception as e:
        print("==========Error reading user prompts==========\n", e)
        exit()

    response_gen = LlmResGen(llm=LLM(),
                             files_dirs=files_dirs)
    answer = []
    for prompt in prompts:
        answer.append(response_gen.gen_answer_obj(prompt["prompt"]))
        print(answer[-1])
        print()
    answer_json = json.dumps(answer, indent=4)
    file = Path(files_dirs["output"])
    file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(files_dirs["output"], "w") as f:
            f.write(answer_json)
    except Exception as e:
        print("====== Error writing into the output file ======")
        print(e)
        exit()
