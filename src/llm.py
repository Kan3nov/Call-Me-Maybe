from pydantic import BaseModel
from llm_sdk import Small_LLM_Model as LLM


class LlmResGen(BaseModel):
    sys_prompt: str
    llm: LLM

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
        next_token = self.predict_token(prompt)
        func_name = ""
        while (func_name + next_token):
            for function in self.functions:
                if (function.find(func_name + next_token)):
                    break
            else:
                break
            func_name += next_token
        return func_name
