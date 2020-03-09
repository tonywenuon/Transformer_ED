# !/usr/bin/env python
import math
import sys

sys.path.append('../')
from commonly_used_code import config


class IdfAllFact:
    def __init__(self, 
        facts_data_path,
        facts_idf_path,
        global_token_path
        ):
        self.facts_data_path = facts_data_path
        self.facts_idf_path = facts_idf_path

        self.dic_token_idf = dict()
        self.total_docs_count = 0
        self.dic_token = dict()
        self.__get_global_tokens(global_token_path)

    def __get_global_tokens(self, global_token_file):
        with open(global_token_file) as f:
            for line in f:
                elems = line.strip().split('\t')
                self.dic_token[elems[0]] = int(elems[1])

    def calc_idf(self):
        for token, indocs in self.dic_token_idf.items():
            idf = math.log(self.total_docs_count * 1.0 / (1 + indocs), 2)
            self.dic_token_idf[token] = idf

    # save idf file
    def save_idf(self):
        outfile = open(self.facts_idf_path, 'w')
        for token, idf in self.dic_token_idf.items():
            write_line = '\t'.join([str(token), str(idf)])
            write_line += '\n'
            outfile.write(write_line)
        outfile.close()

    def get_idf(self):
        self.total_docs_count = 0
        with open(self.facts_data_path) as facts_file:
            count = 0
            for line in facts_file:
                count += 1
                if count % 10000 == 0:
                    print('current idf index: %s' % str(count))

                elems = line.strip().split('\t')
                if len(elems) < 2:
                    continue
                for fact in elems[1:]:
                    # take whole fact as a doc
                    self.total_docs_count += 1

                    tokens = fact.split(' ')
                    dic_fact_tokens = dict()
                    # delete duplicated token
                    for token in tokens:
                        dic_fact_tokens[token] = 0
                    for token in dic_fact_tokens.keys():
                        #if token in self.dic_token.keys():
                        if self.dic_token.get(token, -1) != -1:
                            self.dic_token_idf[token] = self.dic_token_idf.get(token, 0) + 1
        self.calc_idf()
        self.save_idf()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError('Please provide a parameter: wizard or reddit')
    print('calculating IDF for all train and facts set...')
    tag = sys.argv[1]
    if tag == 'wizard':
        iaf = IdfAllFact(config.wizard_train_facts_path,
                         config.wizard_facts_idf_path,
                         config.wizard_global_token_path
                        )
    elif tag == 'reddit':
        iaf = IdfAllFact(config.reddit_train_facts_path,
                         config.reddit_facts_idf_path,
                         config.reddit_global_token_path
                        )
    else:
        raise ValueError('Please provide a parameter: wizard or reddit')

    iaf.get_idf()


