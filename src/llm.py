from pydantic import BaseModel, model_validator, InstanceOf
from llm_sdk import Small_LLM_Model as LLM
from typing import Any
import json
from enum import Enum


class Param(Enum):
    INT = "NUMBER"
    STR = "STRING"
    BOOL = "BOOLEAN"
    NUMBER = "NUMBER"
    FLOAT = "NUMBER"
    LIST = "LIST"
    DICT = "DICT"


class LlmResGen(BaseModel):
    sys_prompt: str
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
            print(next_token)
            for function in self.funcs:
                if (function["name"].find(func_name + next_token[0]) != -1):
                    func_name = func_name + next_token[0]
                    prompt += next_token[0]
                    break
            else:
                break
        return func_name

    def end_gen(self, token: str, type: str, prompt: str = None):
        if type == "string":
            idx = prompt.rfind('"')
            if prompt[idx - 1] != "\\":
                return 1
            return 0
        if type == "integer" or type == "number" \
                or type == "float" or type == "boolean":
            if ',' in token or "}" in token:
                return 1
            return 0
        return 0

    def predict_params(self, usr_prompt: str, func_name: str) -> str:
        prompt = f"""
        Your role is to \
        determine the parameters of the function \
        corresponding to the user prompt STRICTLY IN JSON \
        format and end the statement. \
        Always put numbers as with a FLOATING POINT in the parameters.

        Prompt: What is the sum of 2 and 3?
        Response: {{\"prompt\":\"What is the sum of 2 and 3?\",\
        \"name\":\"fn_add_numbers\" ,\"parameters\":\
        {{"a":2.0,"b":3.0}}}}\n

        COMPLY with what stated above verbatim.
        Prompt: {usr_prompt}
        Response: \
        {{"prompt":"{usr_prompt}","name":"{func_name}",parameters:{{\
        """

        parameters = "{"
        func = None
        for x in self.funcs:
            print(x)
            if (x["name"] == func_name):
                func = x
        for param in func["parameters"].keys():
            prompt += f'"{param}":'
            parameters += f'"{param}":'
            while (1):
                new_token = self.predict_token(prompt)
                type = func["parameters"][param]["type"]
                if (self.end_gen(new_token[0], type, parameters)):
                    prompt += ","
                    parameters += ","
                    break
                prompt += new_token[0]
                parameters += new_token[0]
        parameters = parameters[:-1] + "}"
        return parameters
