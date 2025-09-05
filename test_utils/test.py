from pathlib import Path
from format_code import folder_to_prompt_string

def test_folder_to_prompt_string():
    folder_path = Path("../expert_src/memory_agent")
    prompt = folder_to_prompt_string([folder_path])
    with open("expert_code.txt", "w") as f:
        f.write(prompt)

test_folder_to_prompt_string()