try:
    from llm_sdk import Small_LLM_Model as LLM
except Exception as e:
    print("Import Error,", e)

if __name__ == "__main__":
    llm = LLM()
    id_tensor = llm.encode("hello!").tolist()
    id_list = []
    for x in id_tensor:
        for y in x:
            id_list.append(y)
    print(llm.get_logits_from_input_ids(id_tensor))
