# -*- coding: utf-8 -*-

import json
import base64
import tiktoken
import regex as re

"""
https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/encoder.json
"""

with open('encoder.json', 'r', encoding="utf-8") as file:
    encoder = json.load(file)
with open('vocab.bpe', 'r', encoding="utf-8") as file:
    bpe_data = file.read()

bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
cache = {}

def encode_gpt2(text):
    enc = tiktoken.encoding_for_model("gpt-2")
    # for testindex in text:
    tokenarr = enc.encode(text)
    print(str(text) + "->" + str(tokenarr))
    for tokenindex in tokenarr:
        if enc.decode([tokenindex], "ignore") == "":
            print("    " + str(tokenindex) + "->" + str(enc.decode_bytes([tokenindex])))
        else:
            print("    " + str(tokenindex) + "->" + repr(enc.decode([tokenindex], "ignore")))

# 中文转unicode
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


# print(bytes_to_unicode())
"""
{33: '!', 34: '"', 35: '#', 36: '$', 37: '%', 38: '&', 39: "'", 40: '(', 41: ')', 42: '*', 43: '+', 44: ',', 45: '-', 46: '.', 47: '/', 48: '0', 49: '1', 50: '2', 51: '3', 52: '4', 53: '5', 54: '6', 55: '7', 56: '8', 57: '9', 58: ':', 59: ';', 60: '<', 61: '=', 62: '>', 63: '?', 64: '@', 65: 'A', 66: 'B', 67: 'C', 68: 'D', 69: 'E', 70: 'F', 71: 'G', 72: 'H', 73: 'I', 74: 'J', 75: 'K', 76: 'L', 77: 'M', 78: 'N', 79: 'O', 80: 'P', 81: 'Q', 82: 'R', 83: 'S', 84: 'T', 85: 'U', 86: 'V', 87: 'W', 88: 'X', 89: 'Y', 90: 'Z', 91: '[', 92: '\\', 93: ']', 94: '^', 95: '_', 96: '`', 97: 'a', 98: 'b', 99: 'c', 100: 'd', 101: 'e', 102: 'f', 103: 'g', 104: 'h', 105: 'i', 106: 'j', 107: 'k', 108: 'l', 109: 'm', 110: 'n', 111: 'o', 112: 'p', 113: 'q', 114: 'r', 115: 's', 116: 't', 117: 'u', 118: 'v', 119: 'w', 120: 'x', 121: 'y', 122: 'z', 123: '{', 124: '|', 125: '}', 126: '~', 161: '¡', 162: '¢', 163: '£', 164: '¤', 165: '¥', 166: '¦', 167: '§', 168: '¨', 169: '©', 170: 'ª', 171: '«', 172: '¬', 174: '®', 175: '¯', 176: '°', 177: '±', 178: '²', 179: '³', 180: '´', 181: 'µ', 182: '¶', 183: '·', 184: '¸', 185: '¹', 186: 'º', 187: '»', 188: '¼', 189: '½', 190: '¾', 191: '¿', 192: 'À', 193: 'Á', 194: 'Â', 195: 'Ã', 196: 'Ä', 197: 'Å', 198: 'Æ', 199: 'Ç', 200: 'È', 201: 'É', 202: 'Ê', 203: 'Ë', 204: 'Ì', 205: 'Í', 206: 'Î', 207: 'Ï', 208: 'Ð', 209: 'Ñ', 210: 'Ò', 211: 'Ó', 212: 'Ô', 213: 'Õ', 214: 'Ö', 215: '×', 216: 'Ø', 217: 'Ù', 218: 'Ú', 219: 'Û', 220: 'Ü', 221: 'Ý', 222: 'Þ', 223: 'ß', 224: 'à', 225: 'á', 226: 'â', 227: 'ã', 228: 'ä', 229: 'å', 230: 'æ', 231: 'ç', 232: 'è', 233: 'é', 234: 'ê', 235: 'ë', 236: 'ì', 237: 'í', 238: 'î', 239: 'ï', 240: 'ð', 241: 'ñ', 242: 'ò', 243: 'ó', 244: 'ô', 245: 'õ', 246: 'ö', 247: '÷', 248: 'ø', 249: 'ù', 250: 'ú', 251: 'û', 252: 'ü', 253: 'ý', 254: 'þ', 255: 'ÿ', 0: 'Ā', 1: 'ā', 2: 'Ă', 3: 'ă', 4: 'Ą', 5: 'ą', 6: 'Ć', 7: 'ć', 8: 'Ĉ', 9: 'ĉ', 10: 'Ċ', 11: 'ċ', 12: 'Č', 13: 'č', 14: 'Ď', 15: 'ď', 16: 'Đ', 17: 'đ', 18: 'Ē', 19: 'ē', 20: 'Ĕ', 21: 'ĕ', 22: 'Ė', 23: 'ė', 24: 'Ę', 25: 'ę', 26: 'Ě', 27: 'ě', 28: 'Ĝ', 29: 'ĝ', 30: 'Ğ', 31: 'ğ', 32: 'Ġ', 127: 'ġ', 128: 'Ģ', 129: 'ģ', 130: 'Ĥ', 131: 'ĥ', 132: 'Ħ', 133: 'ħ', 134: 'Ĩ', 135: 'ĩ', 136: 'Ī', 137: 'ī', 138: 'Ĭ', 139: 'ĭ', 140: 'Į', 141: 'į', 142: 'İ', 143: 'ı', 144: 'Ĳ', 145: 'ĳ', 146: 'Ĵ', 147: 'ĵ', 148: 'Ķ', 149: 'ķ', 150: 'ĸ', 151: 'Ĺ', 152: 'ĺ', 153: 'Ļ', 154: 'ļ', 155: 'Ľ', 156: 'ľ', 157: 'Ŀ', 158: 'ŀ', 159: 'Ł', 160: 'ł', 173: 'Ń'}
"""

def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def bpe(token):
    if token in cache:
        return cache[token]
    word = tuple(token)
    pairs = get_pairs(word)

    if not pairs:
        return token

    while True:
        bigram = min(pairs, key=lambda pair: bpe_ranks.get(pair, float('inf')))
        if bigram not in bpe_ranks:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except:
                new_word.extend(word[i:])
                break

            if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)
    word = ' '.join(word)
    cache[token] = word
    return word

def encode(text):
    """
    gpt2 encode实现

    流程 先将输入的文本用正则进行分词(token), 然后将逐个token转utf-8格式(token_utf8), 然后把utf-8格式的数字数组(tokenord)
    逐个转化为unicode格式的(token_byte), 然后把unicode格式的token_byte逐个分解成字母, 然后用已有的合并表合并,
    直到没有可以合并的返回token_byte_bpe, 然后将token_byte_bpe逐个用训练好的编码表映到对应的编码后的tokenid(bpe_tokens)
    :param text:
    :return:
    """
    pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    byte_encoder = bytes_to_unicode()

    bpe_tokens = []
    print(f"text:{text}")
    for token in re.findall(pat, text):
        token_utf8 = token.encode('utf-8')
        token_byte = ''.join(byte_encoder[b] for b in token_utf8)

        tokenord = []
        for b in token_utf8:
            tokenord.append(b)
        token_byte_bpe = bpe(token_byte).split(' ')
        bpe_tokens.extend(encoder[bpe_token] for bpe_token in token_byte_bpe)
        print(f"token:{token:<10} token_utf8:{str(token_utf8):<12} tokenord:{str(tokenord):<40} token_byte:{token_byte:<10} token_byte_bpe:{str(token_byte_bpe):<15} bpe_tokens:{bpe_tokens}")

    return bpe_tokens

# text = "缺陷"
#
# encode_gpt2(text)
# encode(text)
#
#
# text = "Returns list of utf-8 byte"
#
# encode_gpt2(text)
# encode(text)



def is_contain_chinese(word):
    """
    判断字符串是否包含中文字符
    :param word: 字符串
    :return: 布尔值，True表示包含中文，False表示不包含中文
    """
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    match = pattern.search(word)
    return True if match else False

def find_gpt_chinese_token():
    for token in encoder:
        if is_contain_chinese(token):
            print(token)

# find_gpt_chinese_token()

def gpt4_token():
    """
    将gpt4的编码表转化为utf-8格式并写文件保存
    可以看出gpt4含有的中文token大概为840,总token 100256 约0.83%
    gpt2中文token为0,总token 50257
    :return:
    """
    with open('cl100k_base.tiktoken', 'r', encoding="utf-8") as file:
        cl100k_base_lines = file.readlines()
    f_log = open(r'cl100k_base.txt', 'w+', encoding='utf-8')
    f_log.close()
    f_log = open(r'cl100k_base.txt', 'a+', encoding='utf-8')

    for cl100k_base_line in cl100k_base_lines:
        words = cl100k_base_line.split()
        token_base64 = words[0]
        decoded_bytes = base64.b64decode(token_base64)
        try:
            decoded_str = decoded_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                decoded_str = decoded_bytes.decode('latin-1')
            except UnicodeDecodeError:
                # 如果仍然失败,则将解码后的字节数据保留为字节串
                decoded_str = str(decoded_bytes, encoding='utf-8', errors='backslashreplace')
        token_index = words[1]
        f_log.write(repr(decoded_str) + " " + repr(token_index) + "\n")
        if is_contain_chinese(decoded_str):
            print(f'''token_base64:{repr(token_base64):<25} decoded_bytes:{repr(decoded_bytes):<25} decoded_str:{repr(decoded_str):<20} token_index:{str(token_index):<10}''')

    f_log.close()


