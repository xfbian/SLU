# -*- coding:utf-8 -*-
from dataloader.common import InputExample, InputFeatures
from tokenization_nobert import FullTokenizer
from utils.constant import PAD_TOKEN, UNK_TOKEN
import tensorflow as tf
import codecs
import os, sys
import numpy as np
import random


class DataLoader(object):

    def __init__(self, train_file_path, dev_file_path, test_file_path=None, is_shuffle=True):
        self.train_file_path = train_file_path
        self.dev_file_path = dev_file_path
        self.test_file_path = test_file_path
        self.is_shuffle = is_shuffle
        self.train_examples = []
        self.dev_examples = []
        self.test_examples = []

    def _load_raw_data(self, path, name='seq.in'):
        fname = os.path.join(path, name)
        assert os.path.isfile(fname), 'file %s does not exist' % fname
        data_list = []
        with codecs.open(fname, 'r', 'utf-8') as f:
            for line in f:
                u_line = unicode(line.rstrip('\r\n'))
                data_list.append(u_line)
        return data_list

    def get_train_examples(self):
        query_list = self._load_raw_data(self.train_file_path)
        ic_label_list = self._load_raw_data(self.train_file_path, name='label')
        sf_label_list = self._load_raw_data(self.train_file_path, name='seq.out')

        assert len(query_list) == len(ic_label_list), \
            'query_num: %s dose not equal ic_label_num: %s' % (len(query_list), len(ic_label_list))
        assert len(ic_label_list) == len(sf_label_list), \
            'ic_label_num: %s dose not equal sf_label_num: %s' % (len(ic_label_list), len(sf_label_list))

        for idx, query in enumerate(query_list):
            query = query_list[idx]
            ic_label = ic_label_list[idx]
            sf_label = sf_label_list[idx]
            self.train_examples.append(InputExample(guid=idx, query=query, ic_label=ic_label, sf_label=sf_label))

        print 'train_examples num: %s' % len(self.train_examples)
        if self.is_shuffle:
            random.shuffle(self.train_examples)

        return self.train_examples

    def get_dev_examples(self):
        query_list = self._load_raw_data(self.dev_file_path)
        ic_label_list = self._load_raw_data(self.dev_file_path, name='label')
        sf_label_list = self._load_raw_data(self.dev_file_path, name='seq.out')

        assert len(query_list) == len(ic_label_list), \
            'query_num: %s dose not equal ic_label_num: %s' % (len(query_list), len(ic_label_list))
        assert len(ic_label_list) == len(sf_label_list), \
            'ic_label_num: %s dose not equal sf_label_num: %s' % (len(ic_label_list), len(sf_label_list))

        for idx, query in enumerate(query_list):
            query = query_list[idx]
            ic_label = ic_label_list[idx]
            sf_label = sf_label_list[idx]
            self.dev_examples.append(InputExample(guid=idx, query=query, ic_label=ic_label, sf_label=sf_label))

        print 'dev_examples num: %s' % len(self.dev_examples)

        return self.dev_examples

    def get_test_examples(self):
        query_list = self._load_raw_data(self.test_file_path)
        ic_label_list = self._load_raw_data(self.test_file_path, name='label')
        sf_label_list = self._load_raw_data(self.test_file_path, name='seq.out')

        assert len(query_list) == len(ic_label_list), \
            'query_num: %s dose not equal ic_label_num: %s' % (len(query_list), len(ic_label_list))
        assert len(ic_label_list) == len(sf_label_list), \
            'ic_label_num: %s dose not equal sf_label_num: %s' % (len(ic_label_list), len(sf_label_list))

        for idx, query in enumerate(query_list):
            query = query_list[idx]
            ic_label = ic_label_list[idx]
            sf_label = sf_label_list[idx]
            self.test_examples.append(InputExample(guid=idx, query=query, ic_label=ic_label, sf_label=sf_label))

        print 'test_examples num: %s' % len(self.test_examples)

        return self.test_examples


class DataProcess(object):
    def __init__(self, examples, tokenizer, config, mode='train'):
        self.examples = examples
        self.tokenizer = tokenizer
        self.config = config
        self.mode = mode

    def processor(self, example, tokenizer, config, mode='train'):
        query = example.query
        guid = example.guid
        ic_label = example.ic_label
        sf_label = example.sf_label
        max_seq_length = config.max_seq_length

        # convert digit in query to 0
        words = []
        for word in query.split():
            # TODO update python3 to solve this problem
            try:
                if str.isdigit(str(word)):
                    words.append(u'0')
                else:
                    words.append(word)
            except:
                words.append(word)

        # token2id
        input_ids = tokenizer.covert_tokens_to_ids(words)
        sf_label_ids = tokenizer.convert_sf_labels_to_ids(sf_label.split())
        assert len(input_ids) == len(sf_label_ids), \
            'before process, input_ids length: %s dose not equal sf_label_ids length: %s' % \
            (len(input_ids), len(sf_label_ids))

        input_len = len(input_ids)
        input_mask = [1] * input_len

        if input_len < max_seq_length:
            padd_num = max_seq_length - input_len
            padd_ids = tokenizer.covert_tokens_to_ids([PAD_TOKEN] * padd_num)
            sf_padd_ids = tokenizer.convert_sf_labels_to_ids([PAD_TOKEN] * padd_num)

            input_ids.extend(padd_ids)
            sf_label_ids.extend(sf_padd_ids)
            input_mask.extend([0] * padd_num)
        else:
            input_ids = input_ids[:max_seq_length]
            sf_label_ids = sf_label_ids[:max_seq_length]
            input_mask = input_mask[:max_seq_length]

        segment_ids = [0] * max_seq_length
        # ic_label2id
        ic_label_id = tokenizer.convert_ic_labels_to_ids([ic_label])[0]

        assert len(input_ids) == max_seq_length, \
            'after porcess, input_ids length: %s does not equal max_seq_length: %s' % (len(input_ids), max_seq_length)
        assert len(sf_label_ids) == max_seq_length, \
            'after process, sf_label_ids length: %s does not equal max_seq_length: %s' % (
                len(sf_label_ids), max_seq_length)
        assert len(input_mask) == max_seq_length, \
            'after process, input_mask length: %s does not equal max_seq_length: %s' % (len(input_mask), max_seq_length)
        assert len(segment_ids) == max_seq_length, \
            'after process, segment_ids length: %s does not equal max_seq_length: %s' % (
                len(segment_ids), max_seq_length)

        feature = InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                                ic_label_id=ic_label_id, sf_label_ids=sf_label_ids, input_len=input_len, guid=guid)

        return feature

    '''
    InputExample(guid=idx, query=query, ic_label=ic_label, sf_label=sf_label)
    InputFeature()
    
    InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                                ic_label_id=ic_label_id, sf_label_ids=sf_label_ids, input_len=input_len, guid=guid)
    
    '''

    def __call__(self, raw_data_idx, *args, **kwargs):

        idx_example = self.examples[raw_data_idx]
        feature = self.processor(idx_example, self.tokenizer, self.config, self.mode)

        input_ids = np.array(feature.input_ids, dtype=np.int32)
        input_mask = np.array(feature.input_mask, dtype=np.int32)
        segment_ids = np.array(feature.segment_ids, dtype=np.int32)
        ic_label_id = np.array(feature.ic_label_id, dtype=np.int32)
        sf_label_ids = np.array(feature.sf_label_ids, dtype=np.int32)
        input_len = np.array(feature.input_len, dtype=np.int32)
        guid = np.array(feature.guid, dtype=np.int32)

        return input_ids, input_mask, segment_ids, ic_label_id, sf_label_ids, input_len, guid


class DataGenerator(object):
    def __init__(self, data_size):
        self.data_size = data_size

    def __len__(self):
        return self.data_size

    def __call__(self, *args, **kwargs):
        for idx in range(self.data_size):
            raw_data_idx = np.array(idx, dtype=np.int32)
            # yield idx
            # integer scalar arrays can be converted to a scalar index automatically
            yield raw_data_idx


class TFDataSet(object):
    def __init__(self, examples, tokenizer, config, mode='train'):
        self.dataProcess_func = DataProcess(examples, tokenizer, config, mode)
        self.dataGenerator = DataGenerator(len(examples))
        self.batch_size = config.batch_size
        self.max_seq_length = config.max_seq_length
        self.config = config
        self.iterator = self.setDataSetIterator(config, mode)

    def setDataSetIterator(self, config, mode):
        dataset = tf.data.Dataset.from_generator(self.dataGenerator, (tf.int32))
        if config.is_shuffle and mode == 'train':
            dataset = dataset.shuffle(config.data_shuffle_num)

        if mode == 'train':
            dataset = dataset.repeat(config.num_epochs)

        dataset = dataset.map(lambda raw_data: tf.py_func(func=self.dataProcess_func, inp=[raw_data],
                                                          Tout=[tf.int32, tf.int32, tf.int32, tf.int32, tf.int32,
                                                                tf.int32, tf.int32]),
                              num_parallel_calls=config.num_parallel_calls)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(config.data_prefetch_num)
        iterator = dataset.make_initializable_iterator()

        return iterator

    def check_batch_size(self, input_ids, input_mask, input_segment_ids, ic_label_id, sf_label_ids, input_len, guid):

        def f1(): return input_ids, input_mask, input_segment_ids, ic_label_id, sf_label_ids, input_len, guid

        def f2(): return (
            tf.constant(np.zeros([self.batch_size, self.max_seq_length], dtype=np.int32), dtype=tf.int32),
            tf.constant(np.zeros([self.batch_size, self.max_seq_length], dtype=np.int32), dtype=tf.int32),
            tf.constant(np.zeros([self.batch_size, self.max_seq_length], dtype=np.int32), dtype=tf.int32),
            tf.constant(np.zeros([self.batch_size], dtype=np.int32), dtype=tf.int32),
            tf.constant(np.zeros([self.batch_size, self.max_seq_length], dtype=np.int32), dtype=tf.int32),
            tf.constant(np.zeros([self.batch_size], dtype=np.int32), dtype=tf.int32),
            tf.constant(np.zeros([self.batch_size], dtype=np.int32), dtype=tf.int32)

        )

        dim_B = tf.shape(input_ids)[0]
        input_ids, input_mask, input_segment_ids, ic_label_id, sf_label_ids, input_len, guid = tf.cond(
            tf.equal(dim_B, tf.constant(self.batch_size, dtype=tf.int32)), f1, f2)

        input_ids.set_shape([self.batch_size, self.max_seq_length])
        input_mask.set_shape([self.batch_size, self.max_seq_length])
        input_segment_ids.set_shape([self.batch_size, self.max_seq_length])

        ic_label_id.set_shape([self.batch_size])
        ic_label_id = tf.reshape(ic_label_id, [-1])

        sf_label_ids.set_shape([self.batch_size, self.max_seq_length])

        input_len.set_shape([self.batch_size])
        input_len = tf.reshape(input_len, [-1])

        guid.set_shape([self.batch_size])
        guid = tf.reshape(guid, [-1])

        return input_ids, input_mask, input_segment_ids, ic_label_id, sf_label_ids, input_len, guid

    def getBatch(self):
        value = self.iterator.get_next()
        input_ids, input_mask, input_segment_ids, ic_label_id, sf_label_ids, input_len, guid = self.check_batch_size(
            *value)

        return input_ids, input_mask, input_segment_ids, ic_label_id, sf_label_ids, input_len, guid

    def getBatch_multiGPU(self, num_GPU):
        value = self.iterator.get_next()
        input_ids, input_mask, input_segment_ids, ic_label_id, sf_label_ids, input_len, guid = self.check_batch_size(
            *value)

        input_ids_split = tf.split(input_ids, num_GPU)
        input_mask_split = tf.split(input_mask, num_GPU)
        input_segment_ids_split = tf.split(input_segment_ids, num_GPU)
        ic_label_id_split = tf.split(ic_label_id, num_GPU)
        sf_label_ids_split = tf.split(sf_label_ids, num_GPU)
        input_len_split = tf.split(input_len, num_GPU)
        guid_split = tf.split(guid, num_GPU)

        data_batch_multi = []

        for i in range(num_GPU):
            data_batch_multi.append(
                [input_ids_split[i], input_mask_split[i], input_segment_ids_split[i], ic_label_id_split[i],
                 sf_label_ids_split[i], input_len_split[i], guid_split[i]])

        return data_batch_multi
