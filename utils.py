import tensorflow as tf
import unicodedata
import re
import io
import sys

# 下载保存数据集
def downloadData():
    # 获取当前脚本所在路径
    work_path = sys.path[0]

    # 下载并提取文件
    path_to_zip = tf.keras.utils.get_file(
        'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
        extract = True,
        cache_subdir = work_path + "/datasets")
    
    # 提取后的文件路径
    # /Users/zhangyao/Downloads/CSChatbotV3/datasets/spa-eng/spa.txt
    # path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"

# 根据索引获取目录
def getPath(index):
    # 返回数据集目录
    if (index == 1):
        return sys.path[0] + "/datasets/spa-eng/spa.txt"

# 将 unicode 文件转换为 ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

# 文本处理
# 特殊符号处理
# 中文切词
# 增加开始结束token
def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # 在单词与跟在其后的标点符号之间插入一个空格
    # 例如： "he is a boy." => "he is a boy ."
    # 参考：https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # 除了 (a-z, A-Z, ".", "?", "!", ",")，将所有字符替换为空格
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    w = '<start> ' + w + ' <end>'
    return w

# 去除重音符号
# 清理句子
# 返回这样格式的单词对：[目标序列, 输入序列]
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

    return zip(*word_pairs)

# 返回文本最长长度
def max_length(tensor):
    return max(len(t) for t in tensor)

def load_dataset(path, num_examples=None):
    # 创建清理过的输入输出对
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)

    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer
    
# 文本向量（向量）化
def tokenize(lang):
    # 向量化文本(得到序列即单词在字典中下标构成的列表从1开始)
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    # 实现分词（可以得到词频统计结果，词和index的对应关系）
    lang_tokenizer.fit_on_texts(lang)
    # 使用字典将对应词转成index，shape为 (文档数，每条文档的长度)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    # padding 填充位置pre/post truncating 超过maxlen后截取位置pre/post
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')
    # 返回文本和对应的字典序列
    return tensor, lang_tokenizer

# 传入序列和token向量 对应输出显示
def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))