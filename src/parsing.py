try:
    from typing import Any
except Exception as e:
    print("Import error:", e)
    exit()


def validation(argv: list[Any]) -> dict[str, str] | None:
    func_idx = 0
    input_idx = 0
    output_idx = 0
    next = 0
    response = {"functions_definition": "",
                "input": "",
                "output": ""}
    for idx in range(1, len(argv)):
        if (argv[idx] == "--functions_definition"):
            if (idx + 1 >= len(argv)):
                print("Error, option can't be the last argument")
                return None
            func_idx = idx + 1
            next = 1
        elif (argv[idx] == "--input"):
            if (idx + 1 >= len(argv)):
                print("Error, option can't be the last argument")
                return None
            input_idx = idx + 1
            next = 1
        elif (argv[idx] == "--output"):
            if (idx + 1 >= len(argv)):
                print("Error, option can't be the last argument")
                return None
            output_idx = idx + 1
            next = 1
        else:
            if (next):
                next = 0
                continue
            print(f"Error, undefined argument {argv[idx]}")
            return None

    if func_idx != 0:
        response["functions_definition"] = argv[func_idx]
    else:
        response["functions_definition"] = \
            "data/input/functions_definition.json"
    if input_idx != 0:
        response["input"] = argv[input_idx]
    else:
        response["input"] = "data/input/function_calling_tests.json"
    if output_idx != 0:
        response["output"] = argv[output_idx]
    else:
        response["output"] = "data/output/function_calls.json"
    return response
