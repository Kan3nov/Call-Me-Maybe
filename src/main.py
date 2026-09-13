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
        vocabulary = f.read()
        vocab_json = json.loads(vocabulary)

        prompt = 'You have to fill the content with one word and then close the curly bracket' + \
            '{"role": "user", "content": "hello!"}, ' + \
            '{"role": "assistant", "content": '
        response = ""
        next_token = None
        i = 0
        while next_token != 151643 and i < 20:
            id_tensor = llm.encode(prompt)[0].tolist()
            logits = llm.get_logits_from_input_ids(id_tensor)
            next_token = logits.index(max(logits))
            id_tensor.append(next_token)
            response += llm.decode([next_token])
            prompt = llm.decode(id_tensor)
            i += 1
        print("prompt: ", prompt)
        print("response: ", response)
