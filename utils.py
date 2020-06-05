# -*- coding: utf-8 -*-
import copy
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from pypinyin import slug as S ,Style,load_single_dict

from pinyin_split import PinyinSplit


def pad_sents(sents, pad_token):
    """ 根据batch中最长的句子填充其它句子.
    @param sents (list[list[str]]): 句子列表, 每个句子被表示成词序列
    @param pad_token (str): 填充标识
    @returns sents_padded (list[list[str]]): 句子列表，较最长句子短的句子被填充标识符填充
                                            batch 中的句子等长度
    """
    sents_padded = []

    maxLength = max([len(sent) for sent in sents])
    for sentence in sents:
        sents_padded.append(sentence + (maxLength-len(sentence)) * [pad_token]) # 不够长度的用pad_token填充

    return sents_padded



def read_corpus(file_path, source):
    """ 读取文件，每个句子去掉首尾的 '\n' .
    @param file_path (str): 语料库文件
    @param source (str): "tgt" 或 "src" 表示这是src还是tgt 语料库
    """
    data = []
    ps = PinyinSplit()
    # 一字对应一字的拼音
    for line in open(file_path):
        sent = line.strip()
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = list(sent) # 转成列表
            sent = ['<s>'] + sent + ['</s>']
        elif source == 'src':
            sent = ps.split(sent)
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """ 生成源句子和目标句子的batches，长度倒序排序 (最大长度到最小长度).
    @param data (list of (src_sent, tgt_sent)): 元组列表，包含源句子和目标句子
    @param batch_size (int): batch size
    @param shuffle (boolean): 随机洗牌数据集
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

# def hz2pinyin():
#     load_single_dict({ord('嗯'):'en'})
#     with open('./pyin_hz_data/dev.pyin','w') as f:
#         for line in open('./pyin_hz_data/dev.hz'):
#             pinyinList = S('{}'.format(line.strip()),separator='',strict=False)
#             f.write('{}\n'.format(pinyinList))
#     with open('./pyin_hz_data/test.pyin','w') as f:
#         for line in open('./pyin_hz_data/test.hz'):
#             pinyinList = S('{}'.format(line.strip()),separator='',strict=False)
#             f.write('{}\n'.format(pinyinList))
#     with open('./pyin_hz_data/train.pyin','w') as f:
#         for line in open('./pyin_hz_data/train.hz'):
#             pinyinList = S('{}'.format(line.strip()),separator='',strict=False)
#             f.write('{}\n'.format(pinyinList))
#
# def transformData(): # 生语料库转数据集
#     with open('dataset.txt','w') as f:
#         for line in open('people2014.txt'):
#             # sent = line.strip().split('\t')[1].replace(' ','')
#             sent = line.strip().replace(' ', '')
#             if re.search(r'[^\u4e00-\u9fa5\。\；\，\：\“\”\（\）\、\？\《\》\！\…\「\」\【\】]',sent) is not None:
#                 continue
#             f.write('{}\n'.format(sent))
#
# def genDev():
#     with open('pyin_hz_data/dev.hz', 'w') as f:
#         count = 0
#         for line in open('dataset.txt'):
#             count += 1
#             if count % 50 == 36:
#                 f.write('{}'.format(line))
#
# def genTest():
#     with open('pyin_hz_data/test.hz', 'w') as f:
#         count = 0
#         for line in open('dataset.txt'):
#             count += 1
#             if count % 50 == 16:
#                 f.write('{}'.format(line))
#
# def genTrain():
#     with open('pyin_hz_data/train.hz', 'w') as f:
#         count = 0
#         for line in open('dataset.txt'):
#             count += 1
#             if count % 50 != 16 and count % 50!= 36:
#                 f.write('{}'.format(line))

# if __name__ == '__main__':
    # transformData()
    # genDev()
    # genTest()
    # genTrain()
    # hz2pinyin()
    # ps = PinyinSplit()
    # res = ps.split('xiaopuoxiaopu')
    # print(res)
    # print(S('榊何竟呣',separator=''))



