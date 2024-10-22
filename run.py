# -*- coding: utf-8 -*-

"""
Usage:
    run.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE
    run.py decode [options] MODEL_PATH SENTENCE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 64]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 50]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 3]
    --max-num-trial=<int>                   terminate training after how many trials [default: 3]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.0005]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 350]
"""
import math
import sys
import pickle
import time
import os
from collections import Counter


from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nmt_model import Hypothesis, NMT
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm

from pinyin_split import PinyinSplit
from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry

import torch
import torch.nn.utils

def evaluate_ppl(model, dev_data, batch_size=32):
    """ 在验证集上评估困惑度
    @param model (NMT): NMT 模型
    @param dev_data (list of (src_sent, tgt_sent)): 元组列表，包含源句子和目标句子
    @param batch_size (batch size)
    @returns ppl (验证集上的困惑度)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words) # 困惑度=exp(累积损失/累积词数)

    if was_training:
        model.train()

    return ppl


# def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
#     """ Given decoding results and reference sentences, compute corpus-level BLEU score.
#     @param references (List[List[str]]): a list of gold-standard reference target sentences
#     @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
#     @returns bleu_score: corpus-level BLEU score
#     """
#     if references[0][0] == '<s>':
#         references = [ref[1:-1] for ref in references]
#     bleu_score = corpus_bleu([[ref] for ref in references],
#                              [hyp.value for hyp in hypotheses])
#     return bleu_score

def evaluate_ca_hrf(references: List[List[str]], hypotheses: List[Hypothesis]):
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    cum_ca = 0
    sent_num = len(references)
    correct_sent = 0

    for i in range(sent_num):
        ref = Counter(references[i])
        ref_dict = { w:v for w, v in ref.items()} # 统计ref词频
        hyp = Counter(hypotheses[i].value)
        hyp_dict = { w:v for w, v in hyp.items()} # 统计hyp词频

        sum_h = 0 # 分子
        sum_r = 0 # 分母
        for h in hyp_dict:
            if h in ref_dict:
                sum_h += hyp_dict[h] # 统计命中的词数
        for r in ref_dict:
            sum_r += ref_dict[r]
        if sum_h == sum_r: # 完全命中
            correct_sent += 1
        cum_ca += sum_h/sum_r

    avg_ca = cum_ca / sent_num
    hrf = correct_sent / sent_num

    return avg_ca, hrf

def evaluate_khrf(references: List[List[str]],hypotheses: List[List[Hypothesis]]):
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    sent_num = len(references)
    correct_sent = 0
    for i in range(sent_num):
        ref = Counter(references[i])
        ref_dict = {w: v for w, v in ref.items()}  # 统计ref词频
        sum_r = 0
        for r in ref_dict:
            sum_r += ref_dict[r]

        for j in range(len(hypotheses[i])): # k个候选句
            sum_h = 0
            hyp = Counter(hypotheses[i][j].value)
            hyp_dict = {w: v for w, v in hyp.items()}  # 统计hyp词频
            for h in hyp_dict:
                if h in ref_dict:
                    sum_h += hyp_dict[h]  # 统计命中的词数
            if sum_h == sum_r:
                correct_sent += 1
                break # 有一个命中就break，进入下一个句子

    khrf = correct_sent / sent_num
    return khrf


def train(args: Dict):
    """ 训练 NMT 模型.
    @param args (Dict): 命令行参数
    """
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    vocab = Vocab.load(args['--vocab'])

    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab)
    model.train()

    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)
    print(torch.cuda.get_device_name(0))

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr'])) # 优化器

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []

    if os.path.exists(model_save_path):
        # 如果一开始就有模型文件，加载模型（相当于继续未完成的训练）
        print('load previously best model')
        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(params['state_dict'])
        model = model.to(device)

        if os.path.exists(model_save_path+'.optim'):
            print('restore parameters of the optimizers', file=sys.stderr)
            optimizer.load_state_dict(torch.load(model_save_path + '.optim'))
            lr = optimizer.param_groups[0]['lr']
            print('current learning rate is {}'.format(lr))


    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(src_sents)

            example_losses = -model(src_sents, tgt_sents) # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            # 梯度爆炸
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples), file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl
                dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # 降低学习率, 从之前最好的点恢复参数
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # 加载模型
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)

def decode(args: Dict[str, str]):
    """ 在测试集上执行解码操作, 保存最高得分的解码结果.
        如果给定标准句子，函数还会计算平均字符准确率CA，第一个候选句子命中率HRF，前k个候选句子命中率kHRF
    @param args (Dict): 命令行参数
    """
    if args['SENTENCE']:
        ps = PinyinSplit()
        test_data_src = [ps.split(args['SENTENCE'])]
    if args['TEST_SOURCE_FILE']:
        print("load test source sentences from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)
        test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')

    if args['TEST_TARGET_FILE']:
        print("load test target sentences from [{}]".format(args['TEST_TARGET_FILE']), file=sys.stderr)
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)
    model = NMT.load(args['MODEL_PATH'])

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    beam_size = int(args['--beam-size'])
    hypotheses = beam_search(model, test_data_src,
                             beam_size=beam_size,
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses] # 每句话转汉字的首选项形成的列表
        avg_ca,hrf = evaluate_ca_hrf(test_data_tgt,top_hypotheses)
        khrf = evaluate_khrf(test_data_tgt,hypotheses)
        # bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses) # 打分
        # print('Corpus BLEU: {}'.format(bleu_score * 100), file=sys.stderr)
        print('avg_ca: {}'.format(avg_ca),file=sys.stderr)
        print('hrf: {}'.format(hrf), file=sys.stderr)
        print('{}hrf: {}'.format(beam_size,khrf), file=sys.stderr)

    if args['OUTPUT_FILE']:
        with open(args['OUTPUT_FILE'], 'w') as f:
            for src_sent, hyps in zip(test_data_src, hypotheses):
                top_hyp = hyps[0]
                hyp_sent = ''.join(top_hyp.value)
                f.write(hyp_sent + '\n')

    if args['SENTENCE']:
        print('source sentence: {}'.format(args['SENTENCE']))
        for i in range(len(hypotheses[0])):
            result = ''.join(hypotheses[0][i].value)
            print('top_{}_hypotheses_{}: {}'.format(beam_size,i+1,result))


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    """ 对源句子列表使用beam search去构建假设.
    @param model (NMT): NMT 模型
    @param test_data_src (List[List[str]]): 源句子列表, 测试集中的.
    @param beam_size (int): beam_size (每一步的候选数)
    @param max_decoding_time_step (int): Beam search 能产生的最大句子长度
    @returns hypotheses (List[List[Hypothesis]]): 每个源句子的beam_size个假设.
    """
    was_training = model.training
    model.eval()

    hypotheses = [] # 所有句子的候选句列表
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps) # 把这句话的所有候选句加入列表

    if was_training: model.train(was_training)

    return hypotheses

def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # Check pytorch version
    assert(torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version >=1.0.0".format(torch.__version__)

    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()