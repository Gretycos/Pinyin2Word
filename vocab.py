#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    vocab.py --train-src=<file> --train-tgt=<file> [options] VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --train-src=<file>         File of training source sentences
    --train-tgt=<file>         File of training target sentences
    --size=<int>               vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""

from collections import Counter
from docopt import docopt
from itertools import chain
import json
import torch
from typing import List
from utils import read_corpus, pad_sents


class VocabEntry(object):
    """ 词汇表结构
    包含src或者tgt
    """
    def __init__(self, word2id=None):
        """ 初始化词汇表实例
        @param word2id (dict): 文字映射到下标的字典
        """
        if word2id:
            self.word2id = word2id # 词->索引的字典
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0   # Pad Token
            self.word2id['<s>'] = 1 # Start Token
            self.word2id['</s>'] = 2    # End Token
            self.word2id['<unk>'] = 3   # Unknown Token
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()} # 索引->词的字典

    def __getitem__(self, word):
        """ 得到词下标，如果下标不存在则返回<unk>标记
        @param word (str): 需要查询的词
        @returns index (int): 词下标
        """
        return self.word2id.get(word, self.unk_id) # 如果不存在则返回unk token

    def __contains__(self, word):
        """ 检查词是否在词汇表里
        @param word (str): 需要查询的词
        @returns contains (bool): 是否包含在词表
        """
        return word in self.word2id

    def __setitem__(self, key, value):
        """ 如果尝试编辑词表返回异常
        """
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        """ 计算词表的长度
        @returns len (int): 词表的长度
        """
        return len(self.word2id)

    def __repr__(self):
        """ 词表的描述
        """
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        """ 返回下标映射词的结果
        @param wid (int): 词下标
        @returns word (str): 对应的词
        """
        return self.id2word[wid]

    def add(self, word):
        """ 添加没有在词表中的词
        @param word (str): 准备添加到词表里面的词
        @return index (int): 分配好的下标
        """
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        """ 把词序列，句子序列转成下标序列
        @param sents (list[str] or list[list[str]]): 句子列表
        @return word_ids (list[int] or list[list[int]]): 句子下标列表
        """
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        """ 把下标序列转回文字序列
        @param word_ids (list[int]): 词下标序列
        @return sents (list[str]): 词序列
        """
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        """
        把句序列转换成张量，且对较短的句子进行填充
        @param sents (List[List[str]]): 词序列或句序列
        @param device: 用于加载张量的设备，CPU或GPU
        @returns sents_var: 张量，形状(max_sentence_length, batch_size)
        """
        word_ids = self.words2indices(sents) # 获取这个句子对应的词的所有下标
        sents_t = pad_sents(word_ids, self['<pad>']) # 填充标记以对齐
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device) # 张量化
        return torch.t(sents_var)

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        """ 给定语料库，创建词汇表
        @param corpus (list[str]): 从 read_corpus 函数得到的语料库
        @param size (int): 词汇表的词数
        @param freq_cutoff (int): 丢弃的阈值
        @returns vocab_entry (VocabEntry): 从语料库生成的词汇表
        """
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus)) # 获得词频
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff] # 筛选词频大于阈值的词
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), freq_cutoff, len(valid_words)))
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry


class Vocab(object):
    """ 对src和tgt词的总述
    """
    def __init__(self, src_vocab: VocabEntry, tgt_vocab: VocabEntry):
        """ 初始化Vocab
        @param src_vocab (VocabEntry): 源词表
        @param tgt_vocab (VocabEntry): 目标词表
        """
        self.src = src_vocab
        self.tgt = tgt_vocab

    @staticmethod
    def build(src_sents, tgt_sents, vocab_size, freq_cutoff) -> 'Vocab':
        """ 建立词表
        @param src_sents (list[str]): 从 read_corpus() 函数得到的源句子
        @param tgt_sents (list[str]): 从 read_corpus() 函数得到的目标句子
        @param vocab_size (int): 源或目标词表的大小
        @param freq_cutoff (int): 丢弃词的阈值
        """
        assert len(src_sents) == len(tgt_sents)

        print('initialize source vocabulary ..')
        src = VocabEntry.from_corpus(src_sents, vocab_size, freq_cutoff)

        print('initialize target vocabulary ..')
        tgt = VocabEntry.from_corpus(tgt_sents, vocab_size, freq_cutoff)

        return Vocab(src, tgt)

    def save(self, file_path):
        """ 用JSON格式保存词总述
        @param file_path (str): 词总述文件路径
        """
        json.dump(dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id), open(file_path, 'w'), ensure_ascii=False,indent=2)

    @staticmethod
    def load(file_path):
        """ 从JSON格式加载词总述
        @param file_path (str): 词总述文件路径
        @returns 加载好的词总述
        """
        entry = json.load(open(file_path, 'r'))
        src_word2id = entry['src_word2id']
        tgt_word2id = entry['tgt_word2id']

        return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))

    def __repr__(self):
        """ 词总述的描述
        """
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))



if __name__ == '__main__':
    args = docopt(__doc__)

    print('read in source sentences: %s' % args['--train-src'])
    print('read in target sentences: %s' % args['--train-tgt'])

    src_sents = read_corpus(args['--train-src'], source='src')
    tgt_sents = read_corpus(args['--train-tgt'], source='tgt')

    vocab = Vocab.build(src_sents, tgt_sents, int(args['--size']), int(args['--freq-cutoff']))
    print('generated vocabulary, source %d words, target %d words' % (len(vocab.src), len(vocab.tgt)))

    vocab.save(args['VOCAB_FILE'])
    print('vocabulary saved to %s' % args['VOCAB_FILE'])
