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

    流程 先将输入的文本用正则进行分词(token), 然后将逐个token转utf-8格式(token_utf8), 然后把utf-8格式的数字数组(token_utf8)
    逐个转化为unicode格式的字符(token_byte), 然后把unicode格式的token_byte逐个分解成字母, 然后用已有的合并表合并,
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

def get_chinese_chars(input_str):
    # 定义中文Unicode范围
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    # 使用正则表达式查找所有中文字符
    chinese_chars = re.findall(pattern, input_str)
    # 将列表转换为字符串
    chinese_str = ''.join(chinese_chars)
    return chinese_str

def gpt4_token():
    """
    将gpt4的编码表转化为utf-8格式并写文件保存
    可以看出gpt4含有的中文token大概为840,总token 100256 约0.83%
    gpt2中文token为0,总token 50257
    :return:
    """
    chinese_list = []
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
            chinese_list.append(get_chinese_chars(decoded_str))
    f_log.close()
    print(set(chinese_list))
    # {'模', '管', '审', '命周期', '時間', '我', '待', '推', '任务', '别', '日', '提交', '更', '败', '名称', '说明', '型', '联', '字', '径', '已', '手', '志', '小时', '展', '系', '间', '述', '为', '内', '小', '球', '来', '状态', '消', '这', '移', '私', '设', '网', '必', '行', '外', '排序', '笑', '串', '配', '方式', '音', '稿', '此', '内容', '软', '企', '美', '语', '用户名', '排', '备注', '备', '也', '选择', '连接', '北', '产', '在', '所', '序', '价格', '共', '关', '的', '定', '创建', '样', '消息', '解', '条件', '现', '建', '对', '五', '金额', '哈', '邮箱', '机', '位', '歳', '水', '张', '资源', '两', '否', '自治', '条', '和', '问题', '统', '文章', '任', '首', '开', '地', '男', '整', '流', '案', '是', '代码', '或', '广', '県', '从', '去', '义', '播', '亿元', '上传', '队', '考', '器', '项', '向', '思', '称', '额', '运', '认', '尔', '单', '率', '大', '以上', '三', '万', '点击', '计', '下', '雅', '索', '后', '连', '中国', '键', '不', '主', '子', '家', '线', '形', '感', '重', '平', '目', '增', '及', '并', '约', '总', '换', '意', '详情', '提示', '该', '项目', '种', '点', '二', '手机号', '类', '号', '读', '之', '管理员', '完', '发', '添加', '我们', '自', '拉', '删除成功', '用户', '就', '具', '据', '余', '是否', '销', '按钮', '州', '非', '全部', '源', '析', '近', '文件', '监听页面', '族', '服务器', '火', '新增', '以', '等', '注册', '色', '获取', '由', '历', '发布', '数据库', '函数', '素', '話', '阳', '组', '布', '海', '心', '出', '填', '执行', '土', '确认', '界', '周期', '库', '市', '声明', '看', '客', '至', '高', '原', '税', '同时', '错', '商品', '黑', '即', '但', '求', '频', '成', '南', '邮', '验证码', '级', '生命周期函数', '未', '北京', '设计', '稍', '一', '息', '属性', '停', '例', '分类', '番', '今年', '月', '网络', '英', '释', '无', '显示', '监', '科', '监听', '享', '若', '为空', '其中', '情', '输出', '验', '通', '注', '藏', '国', '部', '选', '记', '亿', '段', '場', '期', '异', '雷', '天', '道', '步', '搜索', '们', '辑', '立', '些', '到', '里', '重新', '保存', '分', '开始', '只', '当前', '量', '记录', '最', '微软雅黑', '结', '事件', '有效', '而', '删除', '表', '方', '联系', '制', '注意', '数量', '实', '女', '隐藏', '院', '角', '请', '算', '专', '系统', '板', '支付', '密', '姓名', '如果', '全', '列表', '异常', '知', '店', '动生成', '链接', '路径', '京', '好', '进', '端', '介', '箱', '言', '生命周期', '送', '加', '导', '性', '時', '放', '货', '录', '円', '资', '加载', '比', '関', '供', '元', '象', '页', '改', '方法', '预', '于', '速', '符', '听', '宋体', '效', '相', '不存在', '说', '设计器', '社', '像', '动', '第', '報', '规', '読', '问', '当', '司', '版', '自动生成', '会', '真', '错误', '达', '编', '法', '软雅黑', '见', '需', '正在', '用', '物', '链', '投', '使用', '料', '数字', '标', '始', '电话', '万元', '初始化', '交', '先', '使', '四', '景', '若要', '台', '十', '参', '功能', '份', '老', '构', '作者', '清', '前', '後', '管理', '力', '电', '关闭', '文字', '利', '请输入', '得', '详', '华', '首页', '果', '提', '题', '体', '打', '汽', '表示', '人', '命', '記', '直', '价', '城', '身', '技', '码', '签', '取消', '岁', '您', '确', '试', '县', '再', '游', '通过', '东', '断', '公司', '登录', '中', '标题', '可能', '还', '下午', '版本', '活', '動', '式', '过', '转', '告', '価', '失', '址', '按', '正', '定义', '上', '存', '设置', '生成', '控', '么', '车', '一个', '無料', '束', '支', '指', '环', '起', '视', '回', '户', '写', '反', '证', '不能为空', '检', '開', '需要', '示例', '信息', '程', '完成', '描述', '默认', '陆', '击', '手机', '気', '因', '单位', '给', '验证', '字符', '生', '经', '数', '个', '务', '格式', '发送', '节', '片', '理', '示', '返回', '声', '结束', '简', '明', '请求', '右', '数组', '钟', '确定', '一页', '初', '同', '空', '都', '没有', '处理', '倍', '视频', '收', '不能', '编辑', '请选择', '调', '友', '应', '退', '没', '商', '话', '米', '左', '评', '秒', '列', '安', '省', '記事', '度', '文', '有', '与', '票', '输', '新', '数据', '更新', '权', '何', '能', '传', '对象', '其他', '周', '变', '特', '时间', '在线', '件', '成功', '画', '权限', '产品', '购', '止', '含', '地址', '名', '款', '除', '口', '态', '本', '接', '例如', '其', '查询', '字段', '他', '参数', '者', '失败', '少', '值', '时', '常', '服务', '江', '間', '场', '图', '分钟', '次', '下载', '午', '路', '误', '引', '查', '入', '启', '议', '政', '金', '处', '置', '结果', '容', '头', '事', '合', '多', '分享', '章', '络', '問', '日期', '民', '找', '进行', '配置', '治', '基', '所有', '宋', '影', '大小', '如', '包', '订单', '作', '钮', '登', '存在', '密码', '服', '工', '限', '长', '年', '学', '闭', '评论', '输入', '持', '责', '正确', '载', '册', '账', '付', '来源', '修改', '复', '站', '每', '程序', '测试', '品', '费', '雅黑', '编号', '移到', '取', '退出', '計', '要', '公', '图片', '代', '山', '装', '然', '保', '可以', '無', '易', '集', '造', '了', '化', '门', '报道', '论', '功', '格', '木', '你', '报', '修', '今', '送料', '节点', '类型', '页面', '区', '操作', '核', '长度', '相关', '微', '以下', '员', '星', '业', '信', '超', '局', '見', '我的', '书', '位置', '連', '面', '则', '审核', '优', '询', '字符串', '可', '西', '将'}
    print(len(set(chinese_list)))
    # 762


gpt4_token()