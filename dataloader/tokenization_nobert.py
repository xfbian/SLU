# -*- coding:utf-8 -*-

import os, sys
import codecs
import collections
from utils import constant


def convert_by_vocab(vocab, tokens):
    assert isinstance(vocab, dict), 'vocab param should be dict object'
    output = []
    default = vocab[constant.UNK_TOKEN] if vocab.get(constant.UNK_TOKEN) else None
    for item in tokens:
        output.append(vocab.get(item, default))
    return output


def convert_by_inv_vocab(inv_vocab, ids):
    assert isinstance(inv_vocab, dict), 'inv_vocab param should be dict object'
    output = []
    for item in ids:
        output.append(inv_vocab[item])
    return output

class FullTokenizer(object):
    '''Runs end-to-end tokenization'''

    def __init__(self):
        self.vocab = collections.defaultdict(int)
        self.ic_vocab = collections.defaultdict(int)
        self.sf_vocab = collections.defaultdict(int)
        self.inv_vocab = dict()
        self.inv_ic_vocab = dict()
        self.inv_sf_vocab = dict()

    def create_vocab(self, corpus_file, vocab_out, no_pad=False):
        assert os.path.isfile(corpus_file), 'corpus_file %s does not exist' % corpus_file
        with codecs.open(corpus_file, 'r', 'utf-8') as f:
            for idx, line in enumerate(f):
                u_line = unicode(line.rstrip('\r\n'))
                words = u_line.split()

                for word in words:
                    if word == constant.UNK_TOKEN:
                        break
                    try:
                        # TODO update python3 to solve this problem
                        if str.isdigit(str(word)):
                            word = u'0'
                    except:
                        pass
                    self.vocab[word] += 1

        if no_pad:
            vocab_tmp = [constant.UNK_TOKEN] + sorted(self.vocab, key=self.vocab.get, reverse=True)
        else:
            vocab_tmp = [constant.PAD_TOKEN, constant.UNK_TOKEN] + sorted(self.vocab, key=self.vocab.get, reverse=True)

        with codecs.open(vocab_out, 'w', 'utf-8') as fo:
            for word in vocab_tmp:
                fo.write(word + '\n')

    def create_ic_vocab(self, ic_file, ic_vocab_out):
        assert os.path.isfile(ic_file), 'ic_file %s does not exist' % ic_file
        with codecs.open(ic_file, 'r', 'utf-8') as f:
            for line in f:
                u_line = unicode(line.rstrip('\r\n'))
                words = u_line.split()

                for word in words:
                    self.ic_vocab[word] += 1
        vocab_tmp = sorted(self.ic_vocab, key=self.ic_vocab.get, reverse=True)

        with codecs.open(ic_vocab_out, 'w', 'utf-8') as fo:
            for word in vocab_tmp:
                fo.write(word + '\n')

    def creat_sf_vocab(self, sf_file, sf_vocab_out, no_pad=False):
        assert os.path.isfile(sf_file), 'sf_file %s does not exist' % sf_file
        with codecs.open(sf_file, 'r', 'utf-8') as f:
            for line in f:
                u_line = unicode(line.rstrip('\r\n'))
                words = u_line.split()

                for word in words:
                    self.sf_vocab[word] += 1
        if no_pad:
            vocab_tmp = sorted(self.sf_vocab, key=self.sf_vocab.get, reverse=True)
        else:
            vocab_tmp = [constant.PAD_TOKEN] + sorted(self.sf_vocab, key=self.sf_vocab.get, reverse=True)

        with codecs.open(sf_vocab_out, 'w', 'utf-8') as fo:
            for word in vocab_tmp:
                fo.write(word + '\n')

    def load_vocab(self, vocab_file, tag='vocab'):

        assert os.path.isfile(vocab_file), 'vocab_file %s does not exist' % vocab_file
        vocab_list = []
        with codecs.open(vocab_file, 'r', 'utf-8') as f:
            for line in f:
                u_line = unicode(line.rstrip('\r\n'))
                vocab_list.append(u_line)

        if tag == 'vocab':
            self.vocab = {word: idx for (idx, word) in enumerate(vocab_list)}
            self.inv_vocab = {idx: word for word, idx in self.vocab.items()}
        elif tag == 'ic_vocab':
            self.ic_vocab = {word: idx for (idx, word) in enumerate(vocab_list)}
            self.inv_ic_vocab = {idx: word for word, idx in self.ic_vocab.items()}
        elif tag == 'sf_vocab':
            self.sf_vocab = {word: idx for (idx, word) in enumerate(vocab_list)}
            self.inv_sf_vocab = {idx: word for word, idx in self.sf_vocab.items()}

    def covert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def covert_ids_to_tokens(self, ids):
        return convert_by_inv_vocab(self.inv_vocab, ids)

    def convert_ic_labels_to_ids(self, ic_labels):
        return convert_by_vocab(self.ic_vocab, ic_labels)

    def convert_ids_to_ic_labels(self, ids):
        return convert_by_inv_vocab(self.inv_ic_vocab, ids)

    def convert_sf_labels_to_ids(self, sf_labels):
        return convert_by_vocab(self.sf_vocab, sf_labels)

    def convert_ids_to_sf_labels(self, ids):
        return convert_by_inv_vocab(self.inv_sf_vocab, ids)