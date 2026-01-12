import unicodedata
import requests
import random
import hashlib


def translate_cn_text(text):
    if contains_chinese(text):
        return baidu_translate(text)
    else:
        return text


def contains_chinese(text):
    """检查文本是否包含中文字符"""
    for char in text:
        try:
            name = unicodedata.name(char)
            if 'CJK UNIFIED IDEOGRAPH' in name or 'CJK COMPATIBILITY IDEOGRAPH' in name:
                return True
        except ValueError:
            continue
    return False


def baidu_translate(text):
    """百度翻译API实现（需要自行申请API密钥）"""
    url = 'https://fanyi-api.baidu.com/api/trans/vip/translate'
    salt = random.randint(32768, 65536)
    appid, app_key = "20250317002306121", "B4fHUBrHeeZVFrtVdbDl"
    sign_str = appid + text + str(salt) + app_key
    sign = hashlib.md5(sign_str.encode()).hexdigest()

    params = {
        'q': text,
        'from': 'zh',
        'to': 'en',
        'appid': appid,
        'salt': salt,
        'sign': sign
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    result = response.json()

    if 'error_code' in result:
        raise Exception(f'翻译失败 {result["error_code"]}: {result["error_msg"]}')

    return '\n'.join([res['dst'] for res in result['trans_result']])


# def main():
#     user_input = input("请输入文本：")
#
#     if contains_chinese(user_input):
#         print("检测到中文内容，启动翻译流程...")
#
#         try:
#             translated = baidu_translate(user_input)
#             print(f"\n翻译结果：\n{translated}")
#         except Exception as e:
#             print(f"翻译失败：{str(e)}")
#     else:
#         print("未检测到中文内容，无需翻译")
#
#
# if __name__ == "__main__":
#     main()