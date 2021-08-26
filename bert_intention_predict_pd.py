# -*- coding:utf-8 -*-
from bert import tokenization
import tensorflow as tf
import numpy as np
import keybert
from keybert import KeyBERT
import jieba

bert_config_file = './chinese_L-12_H-768_A-12/bert_config.json'
vocab_file = './chinese_L-12_H-768_A-12/vocab.txt'
max_seq_length = 128
model_name = 'TAAI'
# label_list=['週一','週二','週三','週四','週五','週六','週日','其他'] #1to7
labels_list1=['地址','電話','營業時間','問候','商品','交通工具'] #check

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

def convert_single_example(example, label_map, max_seq_length, tokenizer):
    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    label_ids = label_map[example.label]

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids)
    return feature

def main(_):
    predict_fn = tf.contrib.predictor.from_saved_model('./my_model/{}'.format(model_name))


    # label_list=['評論','商家問答','其他'] #level_1_0710
    # label_list=['週一','週二','週三','週四','週五','週六','週日','其他'] #1to7
    label_list=labels_list1
    if '其他' not in label_list:
        label_list.append('其他')

    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    while True:
        query = input("輸入句子> ")
        predict_example = InputExample(guid='text-0', text_a=query, label='其他')
        feature = convert_single_example(predict_example, label_map, max_seq_length, tokenizer)

        prediction = predict_fn({
            "input_ids": [feature.input_ids],
            "input_mask": [feature.input_mask],
            "segment_ids": [feature.segment_ids],
            "label_ids": [feature.label_ids]
        })
        
        jieba.case_sensitive = True
        seg_list = jieba.cut(query)
        seg_list2=" ".join(seg_list)
        model = KeyBERT('bert-base-chinese')
        model.extract_keywords(seg_list2, keyphrase_ngram_range=(1, 1), stop_words='chinese', 
                            use_mmr=True, diversity=0.5, nr_candidates=20)
        keywords = model.extract_keywords(seg_list2)
        index = np.argmax(prediction['probabilities'][0])
        if np.max(prediction['probabilities'][0])<0.8:
            aa="無法辨識"
        else :
            aa=label_list[int(index)]

        print('[{}]属於：{}，得分：{}'.format(query, aa, np.max(prediction['probabilities'][0])))
        print('關鍵詞為：{}'.format(str(keywords)))

if __name__ == "__main__":
    tf.app.run()
