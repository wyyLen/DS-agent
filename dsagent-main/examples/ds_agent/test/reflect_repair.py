import json
import re

from metagpt.logs import logger
from metagpt.utils.common import CodeParser

reflection = """
```json
{
    "reflection": "In the previous implementation, the error occurred due to the incorrect syntax used for outputting the list of outlier countries. The '@' symbol is not valid in this context. Instead, I should simply assign the list to a variable without any special characters. Additionally, I will ensure that the code is structured properly and includes all necessary steps leading up to the output.",
    "improved_impl": "```python\n# Extract the gdpPercap_1982 column\ngdp_1982 = gdp_data['gdpPercap_1982']\n\n# Calculate Q1, Q3, and IQR\nQ1 = gdp_1982.quantile(0.25)\nQ3 = gdp_1982.quantile(0.75)\nIQR = Q3 - Q1\n\n# Define the outlier bounds\nlower_bound = Q1 - 1.5 * IQR\nupper_bound = Q3 + 1.5 * IQR\n\n# Identify outliers\noutlier_countries = gdp_data[(gdp_1982 < lower_bound) | (gdp_1982 > upper_bound)]['country'].tolist()\n\n# Output the list of outlier countries\noutlier_countries = outlier_countries\n```"
}
```
"""


def clean_improved_impl2(text):
    pattern = r"```python(.*?)```"
    cleaned_text = re.sub(pattern, lambda m: m.group(1), text, flags=re.DOTALL)
    return cleaned_text


def clean_improved_impl(reflection):
    if reflection.count('`') > 2:
        start = reflection.find('```json') + len('```json\n')
        end = reflection.rfind('```')
        json_str = reflection[start:end].strip()
        json_str = json_str.replace('```python\n', '').replace('```', '')
        json_str = f"```json\n{json_str}\n```"
        return json_str
    return None


def parse_code(text: str, lang: str = "") -> str:
    pattern = r"```python(.*?)```"
    cleaned_text = re.sub(pattern, lambda m: m.group(1), text, flags=re.DOTALL)
    pattern = rf"```{lang}\s+(.*?)```"
    match = re.search(pattern, cleaned_text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        return repr(code)[1:-1]
    else:
        logger.error(f"{pattern} not match following text:")
        logger.error(cleaned_text)
        return repr(cleaned_text)[1:-1]  # 假设原始文本是代码


rsp = parse_code(reflection, 'json')
json_str = f"```json\n{rsp}\n```"
print(json_str)
# print(rsp)
# d = json.loads(rsp)
# print(d)

# print(clean_improved_impl2(reflection))
# rsp = clean_improved_impl(reflection)
# rsp = CodeParser.parse_code(block=None, text=rsp)
# print(rsp)
# d = json.loads(rsp)
# print(d)

