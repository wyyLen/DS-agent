from llama_index.core import SimpleDirectoryReader
from metagpt.const import EXAMPLE_DATA_PATH

EXP_PATH = EXAMPLE_DATA_PATH / "exp_bank/plan_exp.json"
files = [EXP_PATH]

documents = SimpleDirectoryReader(
    input_files=files,
    required_exts=[".json"],
).load_data()

print(documents)

