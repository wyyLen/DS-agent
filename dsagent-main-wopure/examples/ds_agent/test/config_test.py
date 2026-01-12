from pathlib import Path

from metagpt.config2 import Config
from metagpt.const import METAGPT_ROOT

print("model config:", Config.from_home("config2.yaml"))

gpt4turbo = METAGPT_ROOT / "config" / "gpt-4-turbo.yaml"

print(Config.from_yaml_file(gpt4turbo))

