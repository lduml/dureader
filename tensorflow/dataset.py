# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements data process strategies.
"""

import os
import json
import logging
import numpy as np
from collections import Counter
import pickle
from bert_serving.client import BertClient
from vocab import Vocab
from elmoformanylangs import Embedder


class BRCDataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    max_q_len : max length of question
    max_p_len : max length of passage
    max_p_num : max passage num in one sample
    """
    def __init__(self, max_p_num, max_p_len, max_q_len,
                 train_files=[], dev_files=[], test_files=[], vocab=Vocab(lower=True)):
        self.logger = logging.getLogger("brc")
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        # **************************
        self.do = 1
        # 是否使用ｂｅｒｔ模型得到隐藏层作为词向量输入
        self.do_bert = False
        self.do_elmo = True
        if self.do_bert:
            self.bc = BertClient()
            self.vocab = vocab
            # if train_files:
            #     self._load_batch_size_data_set('train', train_files[0], train=True)
            # if dev_files:
            #     self._load_batch_size_data_set('dev', train_files[0])
            # if test_files:
            #     self._load_batch_size_data_set('test', train_files[0])
        if self.do_elmo:
            self.bc = Embedder('/tmp/ELMoForManyLangs/zhs.model')
            self.vocab = vocab
            # if train_files:
            #     self._load_batch_size_data_set('train', train_files[0], train=True)
            # if dev_files:
            #     self._load_batch_size_data_set('dev', train_files[0])
            # if test_files:
            #     self._load_batch_size_data_set('test', train_files[0])
        # **************************
        self.train_set, self.dev_set, self.test_set = [], [], []
        # list 可以进行相加 +=
        if not self.do_bert and not self.do_elmo:
            print('****************************')
            if train_files:
                # 把训练集的内容放到 self.train_set
                for train_file in train_files:
                    self.train_set += self._load_dataset(train_file, train=True)
                self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))

            if dev_files:
                for dev_file in dev_files:
                    self.dev_set += self._load_dataset(dev_file)
                self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))

            if test_files:
                for test_file in test_files:
                    self.test_set += self._load_dataset(test_file)
                self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))

    def _load_dataset(self, data_path, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        with open(data_path,encoding='UTF-8') as fin:
            data_set = []
            # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
            # 同时列出数据和数据下标，一般用在 for 循环当中
            # 如：>>>seasons = ['Spring', 'Summer', 'Fall', 'Winter']
            #    >>> list(enumerate(seasons))
            #    [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
            # 这里lidx 无实际意义
            for lidx, line in enumerate(fin):
                sample = json.loads(line.strip())
                # 如果是训练集
                if train:
                    if len(sample['answer_spans']) == 0:
                        continue
                    # 待定-----------------------------略去passage长度为500的
                    if sample['answer_spans'][0][1] >= self.max_p_len:
                        continue

                if 'answer_docs' in sample:
                    sample['answer_passages'] = sample['answer_docs']

                sample['question_tokens'] = sample['segmented_question']

                sample['passages'] = []
                for d_idx, doc in enumerate(sample['documents']):
                    # 如果是训练集
                    if train:
                        most_related_para = doc['most_related_para']
                        # passages {passage_tokens, is_selected} # 把最相关的文章段落内容放到passage_tokens中
                        sample['passages'].append(
                            {'passage_tokens': doc['segmented_paragraphs'][most_related_para],
                             'is_selected': doc['is_selected']}
                        )
                    else:
                        para_infos = []
                        for para_tokens in doc['segmented_paragraphs']:
                            question_tokens = sample['segmented_question']
                            # Counter（计数器）是对字典的补充，用于追踪值的出现次数。
                            common_with_question = Counter(para_tokens) & Counter(question_tokens)
                            correct_preds = sum(common_with_question.values())
                            if correct_preds == 0:
                                recall_wrt_question = 0
                            else:
                                recall_wrt_question = float(correct_preds) / len(question_tokens)
                            para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))
                        para_infos.sort(key=lambda x: (-x[1], x[2]))
                        fake_passage_tokens = []
                        for para_info in para_infos[:1]:
                            fake_passage_tokens += para_info[0]
                        sample['passages'].append({'passage_tokens': fake_passage_tokens})
                data_set.append(sample)
        return data_set

    def _one_mini_batch(self, data, indices, pad_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_token_ids': [],
                      'question_length': [],
                      'passage_token_ids': [],
                      'passage_length': [],
                      'start_id': [],
                      'end_id': []}
        max_passage_num = max([len(sample['passages']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)
        for sidx, sample in enumerate(batch_data['raw_data']):
            for pidx in range(max_passage_num):
                if pidx < len(sample['passages']):
                    # 把一个问题对应的几篇文章的ｔoken_ids加一起
                    batch_data['question_token_ids'].append(sample['question_token_ids'])
                    batch_data['question_length'].append(len(sample['question_token_ids']))
                    passage_token_ids = sample['passages'][pidx]['passage_token_ids']
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))
                else:
                    batch_data['question_token_ids'].append([])
                    batch_data['question_length'].append(0)
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_length'].append(0)
        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id)
        for sample in batch_data['raw_data']:
            if 'answer_passages' in sample and len(sample['answer_passages']):
                gold_passage_offset = padded_p_len * sample['answer_passages'][0]
                batch_data['start_id'].append(gold_passage_offset + sample['answer_spans'][0][0])
                batch_data['end_id'].append(gold_passage_offset + sample['answer_spans'][0][1])
            else:
                # fake span for some samples, only valid for testing
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
        if self.do==2:
            self.do = self.do + 1
            print(len(batch_data['question_token_ids']))
            print(len(batch_data['question_token_ids'][0]))
            print((batch_data['question_token_ids']))
            print((batch_data['question_token_ids'][0]))
            print('****************************')
            print(len(batch_data['passage_token_ids']))
            print(len(batch_data['passage_token_ids'][0]))
            print((batch_data['passage_token_ids']))
            print((batch_data['passage_token_ids'][0]))
        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        return batch_data, pad_p_len, pad_q_len

    def save_set_file(self, set_file_to_save, file_dir, file_name):
        with open(os.path.join(file_dir, file_name), 'wb') as fout:
            pickle.dump(set_file_to_save, fout)

    # 返回word迭代器 提取question_tokens、passage_tokens 中所有的字词
    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['question_tokens']:
                    token = token.replace(' ','').replace('　','')
                    if token == None or token == '':
                        print('1')
                        continue 
                    yield token
                for passage in sample['passages']:
                    for token in passage['passage_tokens']:
                        token = token.replace(' ', '').replace('　', '')
                        if token == None or token == '':
                            print('2')
                            continue
                        yield token

    # 将原始 问题以及答案 转为 id编号
    # 存放在 sample['question_token_ids'] passage['passage_token_ids']
    def convert_to_ids(self, vocab):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_token_ids'] = vocab.convert_to_ids(sample['question_tokens'])
                for passage in sample['passages']:
                    passage['passage_token_ids'] = vocab.convert_to_ids(passage['passage_tokens'])

    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id)

    # **************************************************************************
    def _save_one_mini_batch(self, data, indices, pad_id):

        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_token_ids': [],
                      'question_length': [],
                      'passage_token_ids': [],
                      'passage_length': [],
                      'start_id': [],
                      'end_id': []}
        max_passage_num = max([len(sample['passages']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)
        for sidx, sample in enumerate(batch_data['raw_data']):
            for pidx in range(max_passage_num):
                if pidx < len(sample['passages']):
                    # 把一个问题对应的几篇文章的ｔoken_ids加一起
                    batch_data['question_token_ids'].append(sample['question_token_ids'])
                    batch_data['question_length'].append(len(sample['question_token_ids']))
                    passage_token_ids = sample['passages'][pidx]['passage_token_ids']
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))
                else:
                    batch_data['question_token_ids'].append([])
                    batch_data['question_length'].append(0)
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_length'].append(0)
        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id)
        for sample in batch_data['raw_data']:
            if 'answer_passages' in sample and len(sample['answer_passages']):
                gold_passage_offset = padded_p_len * sample['answer_passages'][0]
                batch_data['start_id'].append(gold_passage_offset + sample['answer_spans'][0][0])
                batch_data['end_id'].append(gold_passage_offset + sample['answer_spans'][0][1])
            else:
                # fake span for some samples, only valid for testing
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
        # 数据转换完毕，有start_id和end_id
        # 对　passage_token_ids、question_token_ids
        # 先有ｉｄ转回ｔｏｋｅｎ，再有ｔｏｋｅｎ输到ｂｅｒｔｍｏｄｅｌ，生成向量
        question_token_ids = []
        i = 0
        # 160的长度 160个问题
        for question_token_id in batch_data['question_token_ids']:
            i = i + 1
            # self.logger.info('***************question_token_id : ' + str(i))
            question_token = self.vocab.recover_from_ids(question_token_id)
            # itema = ''
            # for c in question_token:
            #     itema = itema + c
            # sents2elmo
            if self.do_elmo:
                # print(question_token)
                # print(self.bc.sents2elmo(question_token).shape)
                # print(self.bc.sents2elmo(question_token)[0].shape)
                question_token_ids.append(np.array([i.tolist()[0] for i in self.bc.sents2elmo(question_token)]).tolist())
                # question_token_ids.append(self.bc.sents2elmo(question_token))
            if self.do_bert:
                question_token_ids.append(self.bc.encode(question_token))
        self.logger.info('***************question_token_id : ' + str(i))
        batch_data['question_token_ids'] = question_token_ids

        passage_token_ids = []
        i = 0
        for passage_token_id in batch_data['passage_token_ids']:
            i = i + 1
            # self.logger.info('***************passage_token_id : ' + str(i))
            passage_token = self.vocab.recover_from_ids(passage_token_id)
            # itema = ''
            # for c in passage_token:
            #     itema = itema + c
            # *******************************************
            # In [4]: res = bc.encode(['3', '位', '滚轮', '密码锁', '咋', '开', '<blank>', '<blank>', '<blank>'])
            #
            # In [5]: res.shape
            # Out[5]: (9, 768)
            # *****************************************
            # passage_token = ['3', '位', '滚轮', '密码锁', '咋', '开', '<blank>', '<blank>', '<blank>']
            # res = self.bc.sents2elmo(passage_token)
            # numpy_array = np.array([i.tolist()[0] for i in res])
            if self.do_elmo:
                passage_token_ids.append(np.array([i.tolist()[0] for i in self.bc.sents2elmo(passage_token)]).tolist())
                #passage_token_ids.append(self.bc.sents2elmo(passage_token).tolist())
            if self.do_bert:
                passage_token_ids.append(self.bc.encode(passage_token).tolist())
        batch_data['passage_token_ids'] = passage_token_ids
        self.logger.info('***************passage_token_id : ' + str(i))
        return batch_data

    def _save_mini_batch(self, data, batch_size, pad_id, shuffle=True):
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            print('batch_indices', batch_indices)
            return self._save_one_mini_batch(data, batch_indices, pad_id)

    def _sample_to_dataset(self, data_set, train):
        new_data_set = []
        # *转ｉｄ*************************************************************************
        for li, sample in enumerate(data_set):
            # print('convert-id ',li)
            sample['question_token_ids'] = self.vocab.convert_to_ids(sample['segmented_question'])
            # *转ｉｄ*************************************************************************
            # for passage in sample['passages']:
            #     passage['passage_token_ids'] = self.vocab.convert_to_ids(passage['passage_tokens'])
        for lidx, sample in enumerate(data_set):
            # print('lidx: * ',lidx)
            if 'answer_docs' in sample:
                sample['answer_passages'] = sample['answer_docs']

            sample['question_tokens'] = sample['segmented_question']

            sample['passages'] = []
            for d_idx, doc in enumerate(sample['documents']):
                # print('d_idx: * ', d_idx)
                # 如果是训练集
                if train:
                    most_related_para = doc['most_related_para']
                    # passages {passage_tokens, is_selected} # 把最相关的文章段落内容放到passage_tokens中
                    sample['passages'].append(
                        {'passage_tokens': doc['segmented_paragraphs'][most_related_para],
                         'is_selected': doc['is_selected']}
                    )
                else:
                    para_infos = []
                    for para_tokens in doc['segmented_paragraphs']:
                        question_tokens = sample['segmented_question']
                        # Counter（计数器）是对字典的补充，用于追踪值的出现次数。
                        common_with_question = Counter(para_tokens) & Counter(question_tokens)
                        correct_preds = sum(common_with_question.values())
                        if correct_preds == 0:
                            recall_wrt_question = 0
                        else:
                            recall_wrt_question = float(correct_preds) / len(question_tokens)
                        para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))
                    para_infos.sort(key=lambda x: (-x[1], x[2]))
                    fake_passage_tokens = []
                    for para_info in para_infos[:1]:
                        fake_passage_tokens += para_info[0]
                    sample['passages'].append({'passage_tokens': fake_passage_tokens})
            # *转ｉｄ*************************************************************************
            for passage in sample['passages']:
                passage['passage_token_ids'] = self.vocab.convert_to_ids(passage['passage_tokens'])
            # *转ｉｄ*************************************************************************
            new_data_set.append(sample)

        return new_data_set

    # 目前遇到问题　１６　个batch保存下来需要６００MB－１.3GB磁盘占用
    # bert模型服务启动后需要占用５ＧＢ的显存
    # 可以尝试使用ｃｐｕ做服务，一边读取，一边训练
    def load_batch_size_data_set(self, line_number,  data_path, train=False):
        '''
        :param line_number: 从第几行开始读
        :param epoch:
        :param data_path: 路径
        :param train: 是否是训练集
        :return: feed_dict,lidx 返回一个batch 16条数据，以及读取到第几行
        '''
        with open(data_path, encoding='UTF-8') as fin:
            data_set = []
            # 保存到'./'+set_name+'.json'
            for lidx, line in enumerate(fin, 1):
                # self.logger.info('***************lidx : ' + str(lidx))
                # self.logger.info('***************epoch : ' + str(epoch))
                # self.logger.info('***************lidx : ' + str(lidx))
                if lidx < line_number:
                    continue
                sample = json.loads(line.strip())
                # if train:
                #     if len(sample['answer_spans']) == 0:
                #         continue
                #     if sample['answer_spans'][0][1] >= self.max_p_len:
                #         continue
                data_set.append(sample)
                # 加载了16行有效数据后
                if len(data_set) % 16 == 0 and len(data_set) > 0:
                    print('16')
                    # 数据转化为ｉｄ
                    #　转换为ｂａｔｃｈ大小的ｄａｔａｓｅｔ
                    _data_set = self._sample_to_dataset(data_set,train)
                    data_set = []
                    batch_size = 16
                    pad_id = 0
                    print('_save_mini_batch')
                    feed_dict = self._save_mini_batch(_data_set, batch_size, pad_id, shuffle=True)
                    # 保存　
                    # 返回一个batch 16条数据，以及读取到第几行
                    return feed_dict,lidx
                        # fout.write(json.dumps(feed_dict, ensure_ascii=False) + '\n')
                        # 保存完以后清空

