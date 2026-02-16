import re
from tokenizer import Tokenizer

# 给定的文本
text = "a man of few words, but when he speaks, you listen. He is a master of his craft, and he brings a level of intensity to the role that is unmatched. The action sequences are breathtaking, and the story is engaging. I would highly recommend this film to anyone who enjoys a good thriller."

# 统计单词数（简单方法：按空格分割并去除标点）
def count_words(text):
    # 使用正则表达式匹配单词（包括带连字符的单词）
    words = re.findall(r'\b\w+\b', text)
    return len(words)

# 统计token数
def count_tokens(text):
    try:
        tokenizer = Tokenizer()
        # 不添加bos和eos标记，只计算文本本身的token数
        tokens = tokenizer.encode(text, bos=False, eos=False)
        return len(tokens)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return 0

# 执行统计
word_count = count_words(text)
token_count = count_tokens(text)

print(f"文本: {text}")
print(f"单词数: {word_count}")
print(f"Token数: {token_count}")
