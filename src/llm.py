from pydantic import BaseModel, model_validator, InstanceOf
from llm_sdk import Small_LLM_Model as LLM
from typing import Any
import json


class LlmResGen(BaseModel):
    llm: InstanceOf[LLM]
    files_dirs: dict[str, str]
    funcs: list[dict[str, Any]] = None
    funcs_str: str = None
    vocab: list[dict[int, str]] = None

    @model_validator(mode='after')
    def meow(self) -> dict:
        try:
            with open(self.llm.get_path_to_vocab_file(), "r") as f:
                self.vocab = json.loads(f.read())
        except Exception as e:
            print("Error reading vocabulary file:", e)
            exit()
        try:
            with open(self.files_dirs["functions_definition"], "r") as f:
                self.funcs = json.loads(f.read())
            with open(self.files_dirs["functions_definition"], "r") as f:
                self.funcs_str = f.read()
            undefined_func = {"name": "fn_undefined",
                              "description": "used when no function " +
                              "satisfies the prompt",
                              "parameters": {"None": "None"},
                              "returns": {"type": "None"}}
            self.funcs.append(undefined_func)
        except Exception as e:
            print("Error reading function definitions file:", e)
            exit()
        return self

    def gen_answer_obj(self, prompt: str):
        obj = {}
        obj["prompt"] = prompt
        obj["name"] = self.predict_func(prompt)
        params = self.predict_params(prompt, obj["name"])
        obj["parameters"] = params
        return obj

    def predict_token(self, prompt: str) -> tuple[str, int]:
        id_tensor = self.llm.encode(prompt)[0].tolist()
        logits = self.llm.get_logits_from_input_ids(id_tensor)
        token_id = logits.index(max(logits))
        token = self.llm.decode([token_id])
        return (token, token_id)

    def predict_func(self, usr_prompt: str) -> str:
        prompt = f"""\
Your role is to pick the most suitable function \
out of those functions:\
{self.funcs_str}\n that returns right answer to the user \
prompt given according to its description.

Prompt: What is the sum of 2 and 3?
Response: "name":"fn_add_numbers"

COMPLY with what stated above verbatim. \
WRITE the full name of the function.

Prompt: {usr_prompt}
Response: "name":"\
"""
        func_name = ""
        while (1):
            next_token = self.predict_token(prompt)
            for function in self.funcs:
                if (function["name"].find(func_name + next_token[0]) != -1):
                    func_name = func_name + next_token[0]
                    prompt += next_token[0]
                    break
            else:
                break
        return func_name

    def end_gen(self, type: str, param: str):
        if type == "string":
            idx = param.rfind('"')
            if idx > 0 and param[idx - 1] != "\\":
                return 1
            return 0
        elif type == "number" or type == "boolean":
            if ',' in param or "}" in param:
                return 1
            return 0
        elif type == "integer":
            if '.' in param or '}' in param or ',' in param:
                return 1
            return 0
        return 0

    def predict_params(self, usr_prompt: str, func_name: str) -> str:
        prompt = f"""
Your role is to \
determine the parameters of the function \
corresponding to the user prompt. \
Always put numbers with a FLOATING POINT in the parameters. \
Put parameters as they are in the prompt LITERALLY.

Prompt: What is the sum of 2 and 3?
Response: "prompt":"What is the sum of 2 and 3?",\
"name":"fn_add_numbers","parameters":\
{{"a":2.0,"b":3.0}}

COMPLY with what stated above VERBATIM.
Prompt: {usr_prompt}
Response: \
"prompt":"{usr_prompt}","name":"{func_name}",parameters:{{\
"""
        meow = {}
        func = None
        for x in self.funcs:
            if (x["name"] == func_name):
                func = x
        for param in func["parameters"].keys():
            prompt += f'"{param}":'
            parameter = ""
            if (func["parameters"][param]["type"] == "string"):
                prompt += '"'
                parameter += '"'
            prompt_start = len(prompt) - 1
            while (1):
                new_token = self.predict_token(prompt)
                print("new_token : ", new_token)
                type = func["parameters"][param]["type"]
                parameter += new_token[0]
                prompt += new_token[0]
                if (self.end_gen(type, parameter)):
                    if type == "string":
                        for idx, char in enumerate(parameter):
                            if idx == 0:
                                continue
                            elif char == '"' and parameter[idx - 1] != "\\":
                                break
                        meow[param] = parameter[1:idx]
                        prompt = prompt[:prompt_start + idx + 1]
                    elif type == "boolean":
                        char = "," if parameter.find(",") != -1 else "}"
                        idx = parameter.rfind(char)
                        meow[param] = bool(parameter[0:idx].
                                           strip().lower() == "true")
                        idx = prompt.rfind(char)
                        prompt = prompt[:idx]
                    elif type == "number":
                        char = "," if parameter.find(",") != -1 else "}"
                        idx = parameter.rfind(char)
                        meow[param] = float(parameter[0:idx])
                        idx = prompt.rfind(char)
                        prompt = prompt[:idx]
                    elif type == "integer":
                        chars = [".", ",", "}"]
                        idxs = [parameter.rfind(chars[0]),
                                parameter.rfind(chars[1]),
                                parameter.rfind(chars[2])]
                        for i, idx in enumerate(idxs):
                            if idx == -1:
                                idxs[i] = float("inf")
                        idx = min(idxs)
                        char = chars[idxs.index(idx)]
                        meow[param] = int(parameter[0:idx])
                        idx = prompt.rfind(char)
                        prompt = prompt[:idx]
                    prompt += ","
                    break
        return meow
