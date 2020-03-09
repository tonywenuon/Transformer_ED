# !/usr/bin/env python

import sys
import os
import random, queue

sys.path.append('../')
from commonly_used_code import config
from commonly_used_code import helper_fn

class ExtractRedditData:
    def __init__(self, 
                 ori_qa_data_path, 
                 ori_facts_data_path, 
                 train_qa_data_path, 
                 train_fact_data_path, 
                 valid_qa_data_path, 
                 valid_fact_data_path, 
                 test_qa_data_path, 
                 test_fact_data_path, 
                 global_token_data_path=None 
                 ):
        self.QA_UNK_RATE = 0.1 # the upper bound of UNK in question and answer
        self.FACT_MIN_LENGTH = 8
        self.RESPONSE_MIN_LENGTH = 8  # follow: http://workshop.colips.org/dstc7/papers/03.pdf 
        self.RESPONSE_MAX_LENGTH = 30 # follow: arXiv:1702.01932v1
        self.MAX_VOCAB_SIZE = 50000
        self.VALID_RATE = 0.01
        self.TEST_RATE = 0.01
        # input
        self.ori_qa_data_path = ori_qa_data_path
        self.ori_facts_data_path = ori_facts_data_path
                      
        # output
        self.train_qa_data_path = train_qa_data_path
        self.train_fact_data_path = train_fact_data_path
        self.valid_qa_data_path = valid_qa_data_path
        self.valid_fact_data_path = valid_fact_data_path
        self.test_qa_data_path = test_qa_data_path
        self.test_fact_data_path = test_fact_data_path
        # based on train set (qa and fact) to generate global tokens
        self.global_token_data_path = global_token_data_path

        # save the same key between qa and fact
        self.list_key_tag_ques = list()

    def __init_outobjs(self):
        self.dic_qa_objs = dict()
        self.dic_fact_objs = dict()

        self.dic_qa_objs['train'] = open(self.train_qa_data_path, 'w')
        self.dic_qa_objs['valid'] = open(self.valid_qa_data_path, 'w')
        self.dic_qa_objs['test'] = open(self.test_qa_data_path, 'w')
        self.dic_fact_objs['train'] = open(self.train_fact_data_path, 'w')
        self.dic_fact_objs['valid'] = open(self.valid_fact_data_path, 'w')
        self.dic_fact_objs['test'] = open(self.test_fact_data_path, 'w')

    def __close_outobjs(self):
        for key in self.dic_qa_objs.keys():
            self.dic_qa_objs[key].close()
        for key in self.dic_fact_objs.keys():
            self.dic_fact_objs[key].close()

    def __get_cur_outobj(self):
        lower = 1
        upper = 10000
        valid_range = upper * self.VALID_RATE
        test_range = upper * self.TEST_RATE
        valid_upper = valid_range
        test_upper = valid_range + test_range

        num = random.randint(lower, upper)
        if num <= valid_upper:
            return 'valid'
        elif num > valid_upper and num <= test_upper:
            return 'test'
        else:
            return 'train'
        

    # retrict the response length with a certain range
    def __check_whether_kept(self, ques, res):
        if len(res.strip().split(' ')) < self.RESPONSE_MIN_LENGTH:
            return False
        if len(res.strip().split(' ')) > self.RESPONSE_MAX_LENGTH:
            return False
        if len(ques.strip().split(' ')) < self.RESPONSE_MIN_LENGTH:
            return False
        return True

    def __get_qa_data(self):
        def _clean_question_start(ques):
            if ques.find('EOS') != -1:
                # only keep the last turn
                elems = ques.strip().split('EOS')
                ques = elems[-1].strip()
            if ques.startswith('START EOS til :'):
                ques = ques.replace('START EOS til :', '')
                ques = ques.strip()
            if ques.startswith('START EOS til ...'):
                ques = ques.replace('START EOS til ...', '')
                ques = ques.strip()
            if ques.startswith('START EOS ...'):
                ques = ques.replace('START EOS ...', '')
                ques = ques.strip()
            if ques.startswith('START EOS'):
                ques = ques.replace('START EOS', '')
                ques = ques.strip()
            if ques.startswith('START'):
                ques = ques.replace('START', '')
                ques = ques.strip()
            if ques.startswith('EOS'):
                ques = ques.replace('EOS', '')
                ques = ques.strip()
            if ques.startswith('til :'):
                ques = ques.replace('til :', '')
                ques = ques.strip()
            if ques.startswith('til'):
                ques = ques.replace('til', '')
                ques = ques.strip()
            if ques.startswith('...'):
                ques = ques.replace('...', '')
                ques = ques.strip()
            return ques

        with open(self.ori_qa_data_path) as f:
            qa_index = 0
            for line in f:
                if qa_index % 10000 == 0:
                    print('current convos index: %d' % qa_index)
                qa_index += 1

                elems = line.split('\t')
                key = '_'.join([elems[0].strip(), elems[1].strip()])

                ques = elems[-2].strip()
                ques = _clean_question_start(ques)

                res = elems[-1].strip()
                ques = helper_fn.clear_string(ques.strip())
                res = helper_fn.clear_string(res.strip())

                if self.__check_whether_kept(ques, res) == False:
                    continue

                write_line = '\t'.join([str(qa_index), ques, res]) + '\n'
                tag_obj = self.__get_cur_outobj()
                # record current key for which data set
                self.list_key_tag_ques.append((qa_index, key, tag_obj, ques))
                qa_outobj = self.dic_qa_objs[tag_obj]
                qa_outobj.write(write_line)


    def __get_fact_data(self):
        print(self.ori_facts_data_path)
        dic_key_facts = dict()
        with open(self.ori_facts_data_path) as f:
            fact_index = 0
            for line in f:
                if fact_index % 100000 == 0:
                    print('current fact index: %d' % fact_index)
                fact_index += 1
                elems = line.split('\t')

                fact = elems[-1].strip()
                fact = helper_fn.clear_string(fact)
                fact_len = len(fact.split(' '))
                if fact_len < self.FACT_MIN_LENGTH:
                    continue

                key = '_'.join([elems[0].strip(), elems[1].strip()])
                dic_key_facts[key] = dic_key_facts.get(key, list())

                dic_key_facts[key].append(fact)

        # save fact based on key order in qa data
        print('start saving facts...')
        for qa_index, key, tag_obj, ques in self.list_key_tag_ques:
            facts_list = dic_key_facts.get(key, list())
            if len(facts_list) == 0:
                facts_list.append(ques)

            write_line = '\t'.join(facts_list)
            write_line = '\t'.join([str(qa_index), write_line.strip()]) + '\n'
            fact_outobj = self.dic_fact_objs[tag_obj]
            fact_outobj.write(write_line)

    def __generate_global_token(self):
        print('generate global token...')
        outobj = open(self.global_token_data_path, 'w')
        dic_token2index = {}
        dic_tokens_tmp = {}
        index = 0
        for token in config.SPECIAL_TOKENS:
            dic_token2index[token] = index
            index += 1
        print('read ori qa file...')
        with open(self.train_qa_data_path) as f:
            for line in f:
                elems = line.strip().split('\t')
                for i in range(1, len(elems)):
                    cur_sent = elems[i]
                    tokens = cur_sent.split(' ')
                    for token in tokens:
                        token = token.strip()
                        if token == '':
                            continue
                        dic_tokens_tmp[token] = dic_tokens_tmp.get(token, 0) + 1
        print('read ori fact file...')
        with open(self.train_fact_data_path) as f:
            for line in f:
                elems = line.strip().split('\t')
                for i in range(1, len(elems)):
                    cur_sent = elems[i]
                    tokens = cur_sent.split(' ')
                    for token in tokens:
                        token = token.strip()
                        if token == '':
                            continue
                        dic_tokens_tmp[token] = dic_tokens_tmp.get(token, 0) + 1

        pq = queue.PriorityQueue()
        for token in dic_tokens_tmp:
            pq.put((-dic_tokens_tmp[token], token))
        while not pq.empty():
            freq, token = pq.get()
            dic_token2index[token] = index
            index += 1
            if len(dic_token2index) == self.MAX_VOCAB_SIZE:
                break

        for key, index in dic_token2index.items():
            write_line = '\t'.join([key, str(index)])
            write_line += '\n'
            outobj.write(write_line)
        outobj.close()
        print('generate global token done!')
        return dic_token2index

    def __filter_all_files(self):
        print('start filtering the train valid and test sets...')
        dic_token2index = dict()
        def _get_global_tokens():
            print('reading global tokens...')
            dic_t2i = dict()
            with open(self.global_token_data_path) as f:
                for line in f:
                    elems = line.strip().split('\t')
                    token = elems[0].strip()
                    index = elems[1].strip()
                    dic_t2i[token] = index
            return dic_t2i

        # only keep the sentence that UNK tag less than 10%
        def _check_whether_kept(question, target):
            unk_index = dic_token2index[config.UNK_TOKEN]
            q_unk_count = 0
            question_tokens = question.strip().split(' ')
            for token in question_tokens:
                token = token.strip()
                if token == '':
                    continue
                idx = dic_token2index.get(token, unk_index)
                if idx == unk_index:
                    q_unk_count += 1

            a_unk_count = 0
            target_tokens = target.strip().split(' ')
            for token in target_tokens:
                token = token.strip()
                if token == '':
                    continue
                idx = dic_token2index.get(token, unk_index)
                if idx == unk_index:
                    a_unk_count += 1

            ques_rate = q_unk_count * 1.0 / len(question_tokens)
            tar_rate = a_unk_count * 1.0 / len(target_tokens)

            if ques_rate > self.QA_UNK_RATE or tar_rate > self.QA_UNK_RATE:
                return False
            return True 

        # filter data with UNK tag rate (less than 10% UNK tag)
        def _filter_data(qa_data_tmp, fact_data_tmp, qa_outobj, fact_outobj):
            dic_key_qa = dict()
            with open(qa_data_tmp) as f:
                for line in f:
                    elems = line.strip().split('\t')
                    key = elems[0].strip()
                    ques = elems[1].strip()
                    res = elems[2].strip()
                    dic_key_qa[key] = (ques, res)
            with open(fact_data_tmp) as f:
                for line in f:
                    elems = line.strip().split('\t')
                    key = elems[0].strip()
                    ques = dic_key_qa[key][0]
                    res = dic_key_qa[key][1]
                    is_kept = _check_whether_kept(ques, res)
                    if is_kept == True:
                        write_line = '\t'.join([key, ques, res]) + '\n'
                        qa_outobj.write(write_line)
                        fact_outobj.write(line)

        train_qa_tmp = self.train_qa_data_path + '.tmp'
        train_fact_tmp = self.train_fact_data_path + '.tmp'
        valid_qa_tmp = self.valid_qa_data_path + '.tmp'
        valid_fact_tmp = self.valid_fact_data_path + '.tmp'
        test_qa_tmp = self.test_qa_data_path + '.tmp'
        test_fact_tmp = self.test_fact_data_path + '.tmp'
        # if not exist, then rename them
        if not os.path.exists(train_qa_tmp):
            os.rename(self.train_qa_data_path, train_qa_tmp) 
            os.rename(self.train_fact_data_path, train_fact_tmp) 
            os.rename(self.valid_qa_data_path, valid_qa_tmp) 
            os.rename(self.valid_fact_data_path, valid_fact_tmp) 
            os.rename(self.test_qa_data_path, test_qa_tmp) 
            os.rename(self.test_fact_data_path, test_fact_tmp) 

        dic_token2index = _get_global_tokens()
        self.__init_outobjs()
        _filter_data(train_qa_tmp, train_fact_tmp, self.dic_qa_objs['train'], self.dic_fact_objs['train'])
        _filter_data(valid_qa_tmp, valid_fact_tmp, self.dic_qa_objs['valid'], self.dic_fact_objs['valid'])
        _filter_data(test_qa_tmp, test_fact_tmp, self.dic_qa_objs['test'], self.dic_fact_objs['test'])
        self.__close_outobjs()

    def get_reddit_data(self):
        self.__init_outobjs()
        self.__get_qa_data()
        self.__get_fact_data()
        self.__close_outobjs()
        self.__generate_global_token()
        self.__filter_all_files()

if __name__ == '__main__':
    print('Extracting data from Reddit...')
    erd = ExtractRedditData(
                            config.reddit_qa_path,
                            config.reddit_fact_path,
                            config.reddit_train_qa_path,
                            config.reddit_train_facts_path,
                            config.reddit_valid_qa_path,
                            config.reddit_valid_facts_path,
                            config.reddit_test_qa_path,
                            config.reddit_test_facts_path,
                            config.reddit_global_token_path
                           )
    erd.get_reddit_data()

