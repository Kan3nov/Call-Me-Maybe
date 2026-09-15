from pydantic import BaseModel, model_validator, InstanceOf
from llm_sdk import Small_LLM_Model as LLM
from typing import Any
import json


class LlmResGen(BaseModel):
    sys_prompt: str
    llm: InstanceOf[LLM]
    files_dirs: dict[str, str]
    funcs: list[dict[str, Any]] = None
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

    def predict_token(self, prompt: str) -> tuple[str, int]:
        id_tensor = self.llm.encode(prompt)[0].tolist()
        logits = self.llm.get_logits_from_input_ids(id_tensor)
        token_id = logits.index(max(logits))
        token = self.llm.decode([token_id])
        return (token, token_id)

    def predict_func(self, usr_prompt: str) -> str:
        prompt = self.sys_prompt + f"""\
User Prompt: {usr_prompt}\nYour Response: \
{{"prompt":"{usr_prompt}","name":"\
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

    def predict_param(self, func: str) -> str:
        ...
