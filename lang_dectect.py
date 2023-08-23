from langdetect import detect
import re

def identify_language_and_script(text):
    # Detect the language of the text
    detected_language = detect(text)

    # Check if the text contains characters commonly used in Chinese, English, or Japanese scripts
    chinese_chars = re.compile(r'[\u4e00-\u9fff]')
    english_chars = re.compile(r'[A-Za-z]')
    japanese_chars = re.compile(r'[\u3040-\u30FF\u31F0-\u31FF\uFF65-\uFF9F]')

    if chinese_chars.search(text):
        script = "Chinese"
    elif english_chars.search(text):
        script = "English"
    elif japanese_chars.search(text):
        script = "Japanese"
    else:
        script = "Unknown"

    return detected_language, script

# Test the function
text_to_identify = "你好，こんにちは，Hello!"
language, script = identify_language_and_script(text_to_identify)
print(f"Detected Language: {language}")
print(f"Identified Script: {script}")
