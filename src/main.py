try:
    from llm_sdk import Small_LLM_Model as LLM
    import json
except Exception as e:
    print("Import Error,", e)

if __name__ == "__main__":
    llm = LLM()

    vocab_file = llm.get_path_to_vocab_file()
    print(vocab_file)
    with open(vocab_file, "r") as f:
        prompt = "hello!"
        id_tensor = llm.encode(prompt)[0].tolist()
        logits = llm.get_logits_from_input_ids(id_tensor)  # prediction time
        next_token = logits.index(max(logits))
        vocabulary = f.read()
        vocab_json = json.loads(vocabulary)
        for _ in range(10):
            for key, value in vocab_json.items():
                if value == next_token:
                    print(key, end="")
                    break
            prompt += key
            id_tensor = llm.encode(prompt)[0].tolist()
            logits = llm.get_logits_from_input_ids(id_tensor)
            next_token = logits.index(max(logits))
        print()
