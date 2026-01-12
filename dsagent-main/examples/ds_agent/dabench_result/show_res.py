import json
import re

import fire

from metagpt.const import CUSTOM_DA_EVAL_RES_FILE, DI_EVAL_RES_FILE


def show_res(file: str):
    with open(file, 'r') as f:
        data = f.readlines()
    for line in data:
        line_fixed = line.replace("'", '"')
        line_fixed = re.sub(r'\bTrue\b', 'true', line_fixed)
        line_fixed = re.sub(r'\bFalse\b', 'false', line_fixed)
        res = json.loads(line_fixed)
        correctness = res.get('correctness', {})
        # print(correctness)
        ans = True
        for v in correctness.values():
            ans &= v
        print(ans)


if __name__ == '__main__':
    show_res(DI_EVAL_RES_FILE)
