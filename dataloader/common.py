# -*- coding:utf-8 -*-


class InputExample(object):
    def __init__(self, guid, **kwargs):
        '''
        Construct single InputExample
        :param guid: example id, used to identify each example
        :param kwargs: optional dict for param
        '''
        self.guid = guid
        self.kwargs = kwargs
        self.__dict__.update(kwargs)

    def __str__(self):
        print 'guid: %s' % self.guid

        for key, value in self.kwargs.items():
            print '%s=%s' % (key, value)
        return ''


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, ic_label_id, sf_label_ids, **kwargs):
        '''
        Construct single InputFeatures for InputExample
        :param input_ids:
        :param input_mask:
        :param segment_ids:
        :param ic_label_id: label for intent classification task
        :param sf_label_ids: label for slot-filling task
        :param kwargs:
        '''
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.ic_label_id = ic_label_id
        self.sf_label_ids = sf_label_ids
        self.kwargs = kwargs
        self.__dict__.update(kwargs)

    def __str__(self):
        pass
