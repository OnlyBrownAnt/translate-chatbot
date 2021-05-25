from train import train, evaluate
from utils import downloadData

# 翻译
def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

if __name__ == "__main__":
    # 第一次下载数据集
    # downloadData()

    # 训练模型
    # train()

    # 预测
    translate(u'Firme esto Apuntad más alto. ')