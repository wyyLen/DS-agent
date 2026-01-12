import ast
import json
import re

from json_repair import repair_json

from metagpt.logs import logger


def fix_json(json_str: str) -> str:
    # Step 0: 提取大括号包裹的 JSON 内容
    json_match = re.search(r'\{.*}', json_str, flags=re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
    else:
        return json_str
    # Step 1: 替换键名（Key）的单引号为双引号
    key_pattern = r"(?<!\\)'(?=\s*:)"
    json_str = re.sub(key_pattern, '"', json_str)

    # Step 2: 替换值（Value）的单引号为双引号，处理转义单引号
    # 匹配冒号后的任意空格，单引号包裹的内容（允许转义单引号）
    value_pattern = r':\s*\'((?:\\\'|[^\'])*?)\'(?=\s*([,}\]]|$))'
    json_str = re.sub(value_pattern, r': "\1"', json_str)

    # Step 3: 补全缺失的逗号（处理换行或同一行内的键值对）
    # 匹配闭合字符后直接跟随新键的情况
    json_str = re.sub(
        r'(?<=[}\]"0-9])(\s*)(?=\s*")',
        r',\1',
        json_str,
        flags=re.MULTILINE
    )

    # Step 4: 去除末尾多余的逗号
    json_str = re.sub(r',(\s*)(?=[}\]])', r'\1', json_str)

    return json_str


def extract_evaluation_scores(text):
    """
    优化版评分提取方法，使用更可靠的正则表达式模式
    匹配格式示例：
    - Score: 9
    - the score is 9
    - ​**"Thus the correctness score is s"**: 9.0
    """
    # 优化后的正则表达式模式（不使用VERBOSE模式）
    pattern = re.compile(
        r'(?i)(?:score|correctness|rating|value)\b[^\d]*?(\d+\.?\d*)'
    )

    scores = []
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 优先匹配显式的Score行
        if 'score' in line.lower():
            match = pattern.search(line)
            if match:
                try:
                    scores.append(float(match.group(1)))
                    continue  # 已匹配显式Score则跳过后续匹配
                except:
                    pass

        # 泛化匹配其他评分项
        match = pattern.search(line)
        if match:
            try:
                scores.append(float(match.group(1)))
            except ValueError:
                continue

    # 处理评分项顺序和数量
    if len(scores) >= 4:
        return {
            'overall_goal_score': scores[0],
            'code_requirement_score': scores[1],
            'node_prospects_score': scores[2],
            'correctness_score': scores[3]
        }
    elif len(scores) > 0:
        return {'correctness_score': scores[-1]}  # 最后出现的分数作为最终评分
    return None


def extract_final_score(input_str):
    match = re.search(r'correctness score is (\d+(\.\d+)?).', input_str)
    if match:
        return float(match.group(1))
    match2 = re.search(r'correctness score is \*\*(\d+(\.\d+)?)\*\*\.?', input_str)
    if match2:
        return float(match2.group(1))
    sentences = re.split(r'[.!?]', input_str.strip())
    last_sentence = sentences[-1].strip()
    numbers = re.findall(r'\d+(\.\d+)?', last_sentence)
    if numbers:
        return float(max(numbers, key=lambda x: float(x)))
    return None


def extract_last_thought_json(text: str):
    """
    从包含多个Thought的文本中提取最后一个Thought及其后续内容中的JSON数据

    Args:
        text: 原始文本（可能包含多个Thought段落）

    Returns:
        成功时返回解析后的字典，失败返回None
    """
    # 阶段1：定位最后一个Thought的位置
    thought_pattern = r"Thought \d+:"
    matches = list(re.finditer(thought_pattern, text))

    if not matches:
        return text  # 无Thought段落

    # 截取最后一个Thought及其后续内容
    last_thought_start = matches[-1].start()
    truncated_text = text[last_thought_start:]
    return truncated_text


# functions from zhipuAI
def try_parse_json_object(input: str) -> tuple[str, dict]:
    """JSON cleaning and formatting utilities."""
    # Sometimes, the LLM returns a json string with some extra description, this function will clean it up.

    result = None
    try:
        # Try parse first
        result = json.loads(input)
    except json.JSONDecodeError:
        logger.info("Warning: Error decoding faulty json, attempting repair")

    if result:
        return input, result

    _pattern = r"\{(.*)\}"
    _match = re.search(_pattern, input)
    input = "{" + _match.group(1) + "}" if _match else input

    # Clean up json string.
    input = (
        input.replace("{{", "{")
        .replace("}}", "}")
        .replace('"[{', "[{")
        .replace('}]"', "}]")
        .replace("\\", " ")
        .replace("\\n", " ")
        .replace("\n", " ")
        .replace("\r", "")
        .strip()
    )

    # Remove JSON Markdown Frame
    if input.startswith("```"):
        input = input[len("```"):]
    if input.startswith("```json"):
        input = input[len("```json"):]
    if input.endswith("```"):
        input = input[: len(input) - len("```")]

    try:
        result = json.loads(input)
    except json.JSONDecodeError:
        # Fixup potentially malformed json string using json_repair.
        json_info = str(repair_json(json_str=input, return_objects=False))

        # Generate JSON-string output using best-attempt prompting & parsing techniques.
        try:

            if len(json_info) < len(input):
                json_info, result = try_parse_ast_to_json(input)
            else:
                result = json.loads(json_info)

        except json.JSONDecodeError:
            logger.exception("error loading json, json=%s", input)
            return json_info, {}
        else:
            if not isinstance(result, dict):
                logger.exception("not expected dict type. type=%s:", type(result))
                return json_info, {}
            return json_info, result
    else:
        return input, result


def try_parse_ast_to_json(function_string: str) -> tuple[str, dict]:
    """
     # 示例函数字符串
    function_string = "tool_call(first_int={'title': 'First Int', 'type': 'integer'}, second_int={'title': 'Second Int', 'type': 'integer'})"
    :return:
    """

    tree = ast.parse(str(function_string).strip())
    ast_info = ""
    json_result = {}
    # 查找函数调用节点并提取信息
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function_name = node.func.id
            args = {kw.arg: kw.value for kw in node.keywords}
            ast_info += f"Function Name: {function_name}\r\n"
            for arg, value in args.items():
                ast_info += f"Argument Name: {arg}\n"
                ast_info += f"Argument Value: {ast.dump(value)}\n"
                json_result[arg] = ast.literal_eval(value)

    return ast_info, json_result

