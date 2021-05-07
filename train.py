import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import numpy as np
import os
import time

from model import BahdanauAttention, Encoder, Decoder
from utils import max_length, load_dataset, create_dataset, preprocess_sentence, getPath, convert

"""
参数
"""
DATASET_PATH = getPath(1) # 训练数据集路径
BATCH_SIZE = 64 # 一次训练所选取的样本数
embedding_dim = 256 # 词向量维度
units = 1024 # 输出层维度
EPOCHS = 10 # 训练轮数
checkpoint_dir = './model' # 模型检查点路径

"""
定义优化器和损失函数
获取损失值
"""
# 定义优化器
optimizer = tf.keras.optimizers.Adam()
# 定义损失函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

# 获取损失值
def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

"""
通过数据集获取输入序列和目标序列
"""
# 尝试实验不同大小的数据集
# 获得处理后的文本和对应字典序列，例如input_tensor，inp_lang
num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(DATASET_PATH, num_examples)

# 计算目标张量的最大长度 （max_length）
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)

# 采用 80 - 20 的比例切分训练集和验证集 （暂时无用）
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

"""
参数
"""
BUFFER_SIZE = len(input_tensor_train) # 缓冲区大小
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE # 总步数
vocab_inp_size = len(inp_lang.word_index)+1 # 输入字典最大长度
vocab_tar_size = len(targ_lang.word_index)+1 # 目标字典最大长度

"""
构建模型
"""
# 模型编码
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
# 注意力机制
attention_layer = BahdanauAttention(units)
# 模型解码
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

"""
检查点（基于对象保存)
检查是否已经存在检查点 有的话加载最新的检查点
检查点路径 该路径下不能有其他隐藏文件否则做判断时也会为true
（训练和预测的时候都需要做这个初始化）
"""
ckpt = tf.io.gfile.listdir(checkpoint_dir)
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
if ckpt:
  # 加载已存在检查点
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

"""
模型训练
"""
# 单步训练
@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    # 教师强制 - 将目标词作为下一个输入
    for t in range(1, targ.shape[1]):
      # 将编码器输出 （enc_output） 传送至解码器
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += loss_function(targ[:, t], predictions)

      # 使用教师强制
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

# 模型训练
def train():
  """
  创建tf.data数据集
  获取输入向量和目标向量 
  """
  # 加载数据（输入数据的向量，目标数据的向量）
  dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
  # 打乱数据
  dataset = dataset.shuffle(BUFFER_SIZE)
  # 设置 batch size 值
  dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

  """
  进行多轮训练
  """
  for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
      batch_loss = train_step(inp, targ, enc_hidden)
      total_loss += batch_loss

      # 10步1次输出
      if batch % 10 == 0:
          print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                      batch,
                                                      batch_loss.numpy()))
    # 每个周期（epoch），保存（检查点）一次模型
    if (epoch + 1) % 1 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

"""
预测
"""
def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # 存储注意力权重以便后面制图
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # 预测的 ID 被输送回模型
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot