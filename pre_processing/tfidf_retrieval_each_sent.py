# !/usr/bin/env python
import math
import sys
sys.path.append('../')
from commonly_used_code import config

FACTS_NUMBER = 60

def get_tf_vector(sentence):
    dic_sent_tf = dict()
    tokens = sentence.split(' ')
    for token in tokens:
        dic_sent_tf[token] = dic_sent_tf.get(token.strip(), 0) + 1
    return dic_sent_tf

def tfidf_similarity(dic_ques_tfvec, 
                     dic_fact_tfvec, 
                     dic_fact_token_idf
                     ):
    dic_token_idf = dic_fact_token_idf
    if dic_token_idf == -1:
        return 0
    ques_mode = 0
    dic_ques_tfidfvec = dict()
    for token, ques_tf in dic_ques_tfvec.items():
        idf = float(dic_token_idf.get(token.strip(), 0))
        dic_ques_tfidfvec[token.strip()] = ques_tf * idf
        ques_mode += math.pow(ques_tf * idf, 2)

    fact_mode = 0
    dic_fact_tfidfvec = dict()
    for token, fact_tf in dic_fact_tfvec.items():
        idf = float(dic_token_idf.get(token.strip(), 0))
        dic_fact_tfidfvec[token.strip()] = fact_tf * idf
        fact_mode += math.pow(fact_tf * idf, 2)

    sum = 0
    for token, ques_tfidf in dic_ques_tfidfvec.items():
        fact_tfidf = dic_fact_tfidfvec.get(token.strip(), -1)
        if fact_tfidf != -1:
            sum += ques_tfidf * fact_tfidf
    score = sum * 1.0 / (1 + math.sqrt(ques_mode) * math.sqrt(fact_mode))
    return score

def calc_similarity(question, fact_sentences, dic_fact_token_idf):
    dic_fact_index_simiscore = dict()
    # sentence vector
    dic_question_tf_vector = get_tf_vector(question)

    # fact vectors
    for fact_index, fact_content in enumerate(fact_sentences):
        dic_fact_tf_vector = get_tf_vector(fact_content)

        simi_score = tfidf_similarity(dic_question_tf_vector, 
                                      dic_fact_tf_vector, 
                                      dic_fact_token_idf
                                      )
        if simi_score - 0.0 < 0.00000001:
            continue
        dic_fact_index_simiscore[fact_index] = simi_score

    # sort
    sorted_list = sorted(dic_fact_index_simiscore.items(), key=lambda x:x[1], reverse=True)
    return sorted_list[:FACTS_NUMBER]

class RetrievalFacts:
    def __init__(self, 
                       facts_idf_path, 
                       ori_qa_data_path, 
                       ori_facts_data_path, 
                       retrieved_facts_data_path):
        self.FACT_MIN_LENGTH = 10
        # key: fact_key; value: dict(key: token_index; value: idf)
        self.dic_fact_token_idf = dict()

        self.facts_idf_path = facts_idf_path
        self.ori_qa_data_path = ori_qa_data_path
        self.ori_facts_data_path = ori_facts_data_path
        self.retrieved_facts_data_path = retrieved_facts_data_path


    def __get_facts_token_idf(self):
        # read facts token idf
            print('reading facts token and its idf score...')
            with open(self.facts_idf_path) as idf_fact_file:
                for line in idf_fact_file:
                    elems = line.strip().split('\t')
                    token = elems[0].strip()
                    idf = float(elems[1])
                    self.dic_fact_token_idf[token] = idf


    def __get_sent_facts_file(self):
        # read facts file
        print('generate facts file...')
        print(self.retrieved_facts_data_path)
        facts_outobj = open(self.retrieved_facts_data_path, 'w')
        count = 0
        with open(self.ori_facts_data_path) as facts_file:
            for line in facts_file:
                if count % 10000 == 0:
                    print('current index: %s ' % str(count))
                count += 1

                elems = line.strip().split('\t')
                fact_key = elems[0]
                question = self.dic_key_ques[fact_key]

                fact = ' '.join(elems[1:])
                fact_sentences = [] 
                for fact_sent in fact.split('.'):
                    if len(fact_sent.split(' ')) <= self.FACT_MIN_LENGTH:
                        continue
                    fact_sentences.append(fact_sent)

                ### retrieval process
                # fact vector
                res2 = []
                res2.append(str(fact_key))
                sorted_facts_list = calc_similarity(question, 
                                                    fact_sentences, 
                                                    self.dic_fact_token_idf
                                                    )
                dic_unique = {}
                for (fact_index, simi_score) in sorted_facts_list:
                    fact_content = fact_sentences[fact_index].strip()
                    is_contain = dic_unique.get(fact_content, -1)
                    if is_contain == -1:
                        res2.append(fact_content)
                        dic_unique[fact_content] = 1
                # if it has no facts, return itself
                if len(res2) == 1:
                    res2.append(config.NO_FACT)
        
                write_line = '\t'.join(res2)
                write_line = write_line.strip() + '\n'
                facts_outobj.write(write_line)
        facts_outobj.close()


    def __get_question_data(self):
        print('reading question data...')
        print(self.ori_qa_data_path)
        self.dic_key_ques = dict()
        with open(self.ori_qa_data_path) as f:
            for index, line in enumerate(f):
                if index % 10000 == 0:
                    print('current index: %s ' % str(index))

                # key: token_index; value: term frequency
                dic_queston_tf = dict()
                elems = line.strip().split('\t')
                fact_key = elems[0]
                question = elems[1]
                self.dic_key_ques[fact_key] = question


    def get_retrieved_facts(self):
        self.__get_facts_token_idf()
        self.__get_question_data()
        self.__get_sent_facts_file()
        #self.__get_convos_and_retrieval()
    

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError('Please provide a parameter: wizard or reddit')
    tag = sys.argv[1]
    if tag == 'wizard':
        # inputs: 6 files
        facts_idf_path = config.wizard_facts_idf_path

        train_qa_path = config.wizard_train_qa_path
        train_facts_path = config.wizard_train_facts_path
        valid_qa_path = config.wizard_valid_qa_path
        valid_facts_path = config.wizard_valid_facts_path
        test_qa_path = config.wizard_test_qa_path
        test_facts_path = config.wizard_test_facts_path

        # outputs:
        train_sent_fact_path = config.wizard_train_sent_fact_path
        valid_sent_fact_path = config.wizard_valid_sent_fact_path
        test_sent_fact_path = config.wizard_test_sent_fact_path
    elif tag == 'reddit':
        # inputs: 6 files
        facts_idf_path = config.reddit_facts_idf_path

        train_qa_path = config.reddit_train_qa_path
        train_facts_path = config.reddit_train_facts_path
        valid_qa_path = config.reddit_valid_qa_path
        valid_facts_path = config.reddit_valid_facts_path
        test_qa_path = config.reddit_test_qa_path
        test_facts_path = config.reddit_test_facts_path

        # outputs:
        train_sent_fact_path = config.reddit_train_sent_fact_path
        valid_sent_fact_path = config.reddit_valid_sent_fact_path
        test_sent_fact_path = config.reddit_test_sent_fact_path
    else:
        raise ValueError('Please provide a parameter: wizard or reddit')


    print('Retrieving from facts set...')
    rf = RetrievalFacts(
                        facts_idf_path,
                        train_qa_path,
                        train_facts_path,
                        train_sent_fact_path
                        ) 
    rf.get_retrieved_facts()

    rf = RetrievalFacts(
                        facts_idf_path,
                        valid_qa_path, 
                        valid_facts_path, 
                        valid_sent_fact_path)
    rf.get_retrieved_facts()

    rf = RetrievalFacts(
                        facts_idf_path, 
                        test_qa_path, 
                        test_facts_path, 
                        test_sent_fact_path)
    rf.get_retrieved_facts()

