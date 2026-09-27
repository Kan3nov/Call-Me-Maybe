try:
    from pydantic import BaseModel, model_validator, InstanceOf, TypeAdapter
    from llm_sdk import Small_LLM_Model as LLM
    from json_schema import FuncDef
    from typing import Any, Optional
    import json
except Exception as e:
    print("Import error: ", e)
    exit()


class LlmResGen(BaseModel):
    """The JSON answer generator.

        Attributes:
            llm: the model.
            files_dir: input and output files paths.
            funcs: list of function definitions.
            funcs_str: list of function definitions in a string format.
            vocab: The list of all vocabulary provided by the llm.
        """
    llm: InstanceOf[LLM]
    files_dirs: dict[str, str]
    funcs: Optional[list[dict[str, Any]]] = None
    funcs_str: Optional[str] = None
    vocab: Optional[list[dict[int, str]]] = None

    @model_validator(mode='after')
    def meow(self) -> "LlmResGen":
        """Checks whether the input files are existant or not.

        Args:
            self: The LlmResGen object itself.
        Returns:
            LlmResGen: The LlmResGen object itself after validation.
        """
        try:
            with open(self.llm.get_path_to_vocab_file(), "r") as f:
                self.vocab = json.loads(f.read())
        except Exception as e:
            print("Error reading vocabulary file:", e)
            exit()
        try:
            with open(self.files_dirs["functions_definition"], "r") as f:
                content = f.read()
                func_adapter = TypeAdapter(list[FuncDef])
                if (func_adapter.validate_json(content)):
                    self.funcs = json.loads(content)
            undefined_func = {"name": "fn_undefined",
                              "description": "used with irrelevant or empty "
                              "prompts",
                              "parameters": {"none": {"type": "none"}},
                              "returns": {"type": "none"}}
            if (not self.funcs):
                exit()
            self.funcs.append(undefined_func)
            self.funcs_str = json.dumps(self.funcs)
        except Exception as e:
            print("=========Error reading function definitions file=========")
            print(e)
            exit()
        return self

    def gen_answer_obj(self, prompt: str) -> dict[str, Any]:
        """Generates a JSON object that includes the model answer \
        to a user prompt.

        Args:
            prompt: the user prompt.
        Returns:
            dict[str, Any]: the JSON object containing the user prompt, \
            the function name, and the parameters passed to the function.
        """
        obj: dict[str, Any] = {}
        obj["prompt"] = prompt
        obj["name"] = self.predict_func(prompt)
        params = self.predict_params(prompt, obj["name"])
        obj["parameters"] = params
        return obj

    def mask(self, logits: list[float], type: Optional[str]) -> None:
        """Implements masking on the logits produced by the model.

        Args:
            logits: list of integer logits produced by the model.
            type: type of the expected token
        """
        if (type == "integer" or type == "number"):
            chars = "0123456789.,}"
            for token_id in range(len(logits)):
                if set(self.llm.decode([token_id])).isdisjoint(set(chars)):
                    logits[token_id] = float("-inf")
        elif (type == "boolean"):
            chars = "true,false}"
            for token_id in range(len(logits)):
                if set(self.llm.decode([token_id])).isdisjoint(set(chars)):
                    logits[token_id] = float("-inf")
        elif (type == "func"):
            charss: set[str] = set()
            if (not self.funcs):
                return
            for func in self.funcs:
                charss = charss.union(set(func["name"]))
            for token_id in range(len(logits)):
                if set(self.llm.decode([token_id])).isdisjoint(set(charss)):
                    logits[token_id] = float("-inf")

    def predict_token(self, prompt: str,
                      type: Optional[str] = None
                      ) -> tuple[str, int]:
        """Predicts the next token in a prompt.

        Args:
            prompt: the user prompt.
            type: the expected type of the token
        Returns:
            tuple[str, int]: the first element is the token itself, \
            and the second is the token id.
        """
        id_tensor = self.llm.encode(prompt)[0].tolist()
        logits = self.llm.get_logits_from_input_ids(id_tensor)
        self.mask(logits, type)
        token_id = logits.index(max(logits))
        token = self.llm.decode([token_id])
        return (token, token_id)

    def predict_func(self, usr_prompt: str) -> Optional[str]:
        """Predicts the suitable function name for a user prompt.

        Args:
            usr_prompt: the user prompt.
        Returns:
            Optional[str]: the function name; None .
        """
        prompt = f"""\
Your role is to pick a suitable function \
out of those functions ONLY:\
{self.funcs_str}\n that does the demanded action by the user \
prompt. In case of undefined, or emtpy \
prompts; or if no function serves the user's need, \
pick the fn_undefined function. Do not pick a non-existing function.

COMPLY with what stated above verbatim. \
WRITE the full name of the function.

Prompt: "{usr_prompt}",
Response: "function_name":"\
"""
        func_name = ""
        while (1):
            next_token = self.predict_token(prompt, "func")
            if (not self.funcs):
                return None
            for function in self.funcs:
                if (function["name"].find(func_name + next_token[0]) != -1):
                    func_name = func_name + next_token[0]
                    prompt += next_token[0]
                    break
            else:
                break
        return func_name

    def end_gen(self, type: str, param: str) -> int:
        """Determines when to halt the prediction of new tokens.

        Args:
            type: type of the desired parameter.
            param: the already generated parameter.
        Returns:
            int: 1 if prediction should be stopped, 0 if otherwise.
        """
        if type == "string" or type == "none":
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

    def predict_params(self, usr_prompt: str,
                       func_name: str) -> Optional[dict[str, Any]]:
        """Predicts the suitable set of parameters sent to the chosen function.

        Args:
            usr_promt: the user prompt.
            func_name: the function name.
        Returns:
            Optional[dict[str, Any]]: a dictionary of each parameter's name \
            and its corresponding value. Returns None on error.
        """
        if (not self.funcs):
            print("Error predicting function name")
            exit()
        match_funcs = [func for func in self.funcs if dict(func)["name"] ==
                       func_name]
        if (len(match_funcs) == 1):
            func_def = match_funcs[0]
        else:
            print("Error predicting function name")
            exit()
        prompt = f"""
You are an expert AI assistant. Your role is to \
determine the parameters of the function \
corresponding to the user prompt and the function's definition. \
Always put numbers with a FLOATING POINT in the parameters. \
Put parameters as they are in the prompt LITERALLY.

{func_def}

COMPLY with what stated above VERBATIM.
Prompt: {usr_prompt}
Response: \
"prompt":"{usr_prompt}","name":"{func_name}",parameters:{{\
"""
        meow: dict[str, Any] = {}
        func = None
        if (not self.funcs):
            return None
        for x in self.funcs:
            if (x["name"] == func_name):
                func = x
        if (not func):
            return None
        for param in func["parameters"].keys():
            prompt += f'"{param}":'
            parameter = ""
            if (func["parameters"][param]["type"] == "string"):
                prompt += '"'
                parameter += '"'
            prompt_start = len(prompt) - 1
            while (1):
                new_token = self.predict_token(
                                            prompt,
                                            func["parameters"][param]["type"])
                type = func["parameters"][param]["type"]
                parameter += new_token[0]
                prompt += new_token[0]
                if (self.end_gen(type, parameter)):
                    if type == "string" or type == "undefined":
                        parameter = parameter.replace("\\\\", "\\")
                        for idx, char in enumerate(parameter):
                            if idx == 0:
                                continue
                            elif char == '"' and parameter[idx - 1] != "\\":
                                break
                        meow[param] = parameter[1:idx].replace('\\"', '"')
                        prompt = prompt[:prompt_start + idx + 1]
                    elif type == "boolean":
                        char = "," if parameter.find(",") != -1 else "}"
                        idx = parameter.rfind(char)
                        try:
                            meow[param] = bool(parameter[0:idx].
                                               strip().lower() == "true")
                        except Exception as e:
                            print(f"====== Error converting parameter \
                                  {param} to boolean in function \
                                  {func_name} for prompt \
                                  {usr_prompt} ======")
                            print(e)
                        idx = prompt.rfind(char)
                        prompt = prompt[:idx]
                    elif type == "number":
                        char = "," if parameter.find(",") != -1 else "}"
                        idx = parameter.rfind(char)
                        try:
                            meow[param] = float(parameter[0:idx])
                        except Exception as e:
                            print(f"====== Error converting parameter \
                                  {param} to float in function \
                                  {func_name} for prompt \
                                  {usr_prompt} ======")
                            print(e)
                        idx = prompt.rfind(char)
                        prompt = prompt[:idx]
                    elif type == "integer":
                        chars = [".", ",", "}"]
                        idxs: list[int] = [
                                parameter.rfind(chars[0]),
                                parameter.rfind(chars[1]),
                                parameter.rfind(chars[2])]
                        idxs = [idx for idx in idxs if idx != -1]
                        idx = min(idxs)
                        char = chars[idxs.index(idx)]
                        try:
                            meow[param] = int(parameter[0:idx])
                        except Exception as e:
                            print(f"====== Error converting parameter \
                                  {param} to integer in function \
                                  {func_name} for prompt \
                                  {usr_prompt} ======")
                            print(e)
                        idx = prompt.rfind(char)
                        prompt = prompt[:idx]
                    prompt += ","
                    break
        return meow
