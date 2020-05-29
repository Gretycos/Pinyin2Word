# -*- coding: utf-8 -*-
import copy
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from xpinyin import Pinyin

from pinyin_split import PinyinSplit


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    maxLength = max([len(sent) for sent in sents])
    for sentence in sents:
        sents_padded.append(sentence + (maxLength-len(sentence)) * [pad_token]) # 不够长度的用pad_token填充

    return sents_padded



def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
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
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
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

# def parsePinyin(sent):
#     result = []
#     length = len(sent)
#     # 循环匹配对拼音进行分词，假设没有数字和英文
#     while length > 0:
#         word = re.match(r"[b-df-hj-np-tw-z]?h?[iuv]?(ai|ei|ao|ou|er|ang?|eng?|ong|a|o|e|i|u|ng|n)?",sent).group()
#         if len(word) == 0:
#             word = re.match(r"[^A-Za-z]?",sent).group()
#         result.append(word)
#         word_length = len(word)
#         sent = sent[word_length:]
#         length = length - word_length
#     # print(result)
#     return result

def hz2pinyin():
    p = Pinyin()
    with open('./pyin_hz_data/dev.pyin','w') as f:
        for line in open('./pyin_hz_data/dev.hz'):
            pinyin = p.get_pinyin(u'{}'.format(line.strip()),'')
            f.write('{}\n'.format(pinyin))
    with open('./pyin_hz_data/test.pyin','w') as f:
        for line in open('./pyin_hz_data/test.hz'):
            pinyin = p.get_pinyin(u'{}'.format(line.strip()),'')
            f.write('{}\n'.format(pinyin))
    with open('./pyin_hz_data/train.pyin','w') as f:
        for line in open('./pyin_hz_data/train.hz'):
            pinyin = p.get_pinyin(u'{}'.format(line.strip()),'')
            f.write('{}\n'.format(pinyin))

def transformData():
    with open('dataset.txt','w') as f:
        for line in open('zho_news_2007-2009_300K-sentences.txt'):
            sent = line.strip().split('\t')[1].replace(' ','')
            if re.search(r'[^\u4e00-\u9fa5\。\；\，\：\“\”\（\）\、\？\《\》\！\…\「\」\【\】]',sent) is not None:
                continue
            f.write('{}\n'.format(sent))

def genDev():
    with open('pyin_hz_data/dev.hz', 'w') as f:
        count = 0
        for line in open('dataset.txt'):
            count += 1
            if count % 20 == 0:
                f.write('{}'.format(line))

def genTest():
    with open('pyin_hz_data/test.hz', 'w') as f:
        count = 0
        for line in open('dataset.txt'):
            count += 1
            if count % 20 == 1:
                f.write('{}'.format(line))

def genTrain():
    with open('pyin_hz_data/train.hz', 'w') as f:
        count = 0
        for line in open('dataset.txt'):
            count += 1
            if count % 20 != 0 and count % 20!= 1:
                f.write('{}'.format(line))

# if __name__ == '__main__':
    # transformData()
    # genDev()
    # genTest()
    # genTrain()
    # hz2pinyin()
    # ps = PinyinSplit()
    # res = ps.split('tifenereng')
    # print(res)


