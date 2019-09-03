# !/use/bin/env python

import os
import configparser
from .helper_fn import *

parser = configparser.SafeConfigParser()
config_file_path = '../configuration/config.ini'
#config_file_path = 'configuration/config.ini'
parser.read(config_file_path)

# reserve <START> and <END> for source
src_reserved_pos = 2
# reserve <START> or <END> for target
tar_reserved_pos = 1
NO_FACT = 'no_fact'
NO_CONTEXT = 'no_context'
START_TOKEN = '<START>'
END_TOKEN = '<END>'
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
SPECIAL_TOKENS = [START_TOKEN, END_TOKEN, PAD_TOKEN, UNK_TOKEN]

# original data set path
wizard_data_path = parser.get('FilePath', 'wizard_data_path')

train_path = parser.get('FilePath', 'symblic_single_turn_train')
valid_path = parser.get('FilePath', 'symblic_single_turn_valid')
test_path = parser.get('FilePath', 'symblic_single_turn_test')

# wizard data set
stop_words_path = parser.get('Wizard', 'stop_words')
train_data_path = parser.get('Wizard', 'train_data')
valid_data_path = parser.get('Wizard', 'valid_data')
test_data_path = parser.get('Wizard', 'test_data')

# generate symblic question answer and facts
use_for_retrieval_token_path = parser.get('SymblicQAF', 'retrieval_token_dict')
src_global_token_path = parser.get('SymblicQAF', 'src_global_token_dict')
tar_global_token_path = parser.get('SymblicQAF', 'tar_global_token_dict')
pro_qa_data_path = parser.get('SymblicQAF', 'pro_qa_data')
pro_facts_data_path = parser.get('SymblicQAF', 'pro_facts_data')
pro_conv_data_path = parser.get('SymblicQAF', 'pro_conv_data')
sent_fact_data_path = parser.get('SymblicQAF', 'sent_fact_data')
oracle_sent_fact_data_path = parser.get('SymblicQAF', 'oracle_sent_fact_data')

# IDF data path
facts_idf_path = parser.get('TFIDF', 'facts_idf_data')

# wizard train valid test data path
wizard_train_path = os.path.join(wizard_data_path, train_path)
wizard_valid_path = os.path.join(wizard_data_path, valid_path)
wizard_test_path = os.path.join(wizard_data_path, test_path)
makedirs(wizard_train_path)
makedirs(wizard_valid_path)
makedirs(wizard_test_path)

# original data of wizard
wizard_stop_words_path = os.path.join(wizard_data_path, stop_words_path)
wizard_train_data_path = os.path.join(wizard_data_path, train_data_path)
wizard_valid_data_path = os.path.join(wizard_data_path, valid_data_path)
wizard_test_data_path = os.path.join(wizard_data_path, test_data_path)

# used for wizard file path
wizard_use_for_retrieval_token_path = os.path.join(wizard_train_path, use_for_retrieval_token_path)
wizard_global_token_path = os.path.join(wizard_train_path, src_global_token_path)
wizard_facts_idf_path = os.path.join(wizard_train_path, facts_idf_path)

wizard_train_qa_path = os.path.join(wizard_train_path, pro_qa_data_path)
wizard_train_facts_path = os.path.join(wizard_train_path, pro_facts_data_path)
wizard_train_conv_path = os.path.join(wizard_train_path, pro_conv_data_path)
wizard_train_sent_fact_path = os.path.join(wizard_train_path, sent_fact_data_path)
wizard_train_oracle_sent_fact_path = os.path.join(wizard_train_path, oracle_sent_fact_data_path)

wizard_valid_qa_path = os.path.join(wizard_valid_path, pro_qa_data_path)
wizard_valid_facts_path = os.path.join(wizard_valid_path, pro_facts_data_path)
wizard_valid_conv_path = os.path.join(wizard_valid_path, pro_conv_data_path)
wizard_valid_sent_fact_path = os.path.join(wizard_valid_path, sent_fact_data_path)
wizard_valid_oracle_sent_fact_path = os.path.join(wizard_valid_path, oracle_sent_fact_data_path)

wizard_test_qa_path = os.path.join(wizard_test_path, pro_qa_data_path)
wizard_test_facts_path = os.path.join(wizard_test_path, pro_facts_data_path)
wizard_test_conv_path = os.path.join(wizard_test_path, pro_conv_data_path)
wizard_test_sent_fact_path = os.path.join(wizard_test_path, sent_fact_data_path)
wizard_test_oracle_sent_fact_path = os.path.join(wizard_test_path, oracle_sent_fact_data_path)


# for DSTC Reddit
reddit_data_path = parser.get('FilePath', 'reddit_data_path')
train_reddit_fact_path = parser.get('Reddit', 'train_fact_data')
train_reddit_qa_path = parser.get('Reddit', 'train_qa_data')
valid_reddit_fact_path = parser.get('Reddit', 'valid_fact_data')
valid_reddit_qa_path = parser.get('Reddit', 'valid_qa_data')
test_reddit_fact_path = parser.get('Reddit', 'test_fact_data')
test_reddit_qa_path = parser.get('Reddit', 'test_qa_data')

# reddit train valid test data path
reddit_train_path = os.path.join(reddit_data_path, train_path)
reddit_valid_path = os.path.join(reddit_data_path, valid_path)
reddit_test_path = os.path.join(reddit_data_path, test_path)
makedirs(reddit_train_path)
makedirs(reddit_valid_path)
makedirs(reddit_test_path)

# original data of reddit
train_reddit_qa_path = os.path.join(reddit_data_path, train_reddit_qa_path)
train_reddit_fact_path = os.path.join(reddit_data_path, train_reddit_fact_path)
valid_reddit_qa_path = os.path.join(reddit_data_path, valid_reddit_qa_path)
valid_reddit_fact_path = os.path.join(reddit_data_path, valid_reddit_fact_path)
test_reddit_qa_path = os.path.join(reddit_data_path, test_reddit_qa_path)
test_reddit_fact_path = os.path.join(reddit_data_path, test_reddit_fact_path)

# used for reddit file path
reddit_use_for_retrieval_token_path = os.path.join(reddit_train_path, use_for_retrieval_token_path)
reddit_global_token_path = os.path.join(reddit_train_path, src_global_token_path)
reddit_facts_idf_path = os.path.join(reddit_train_path, facts_idf_path)

reddit_train_qa_path = os.path.join(reddit_train_path, pro_qa_data_path)
reddit_train_facts_path = os.path.join(reddit_train_path, pro_facts_data_path)
reddit_train_conv_path = os.path.join(reddit_train_path, pro_conv_data_path)
reddit_train_sent_fact_path = os.path.join(reddit_train_path, sent_fact_data_path)
reddit_train_oracle_sent_fact_path = os.path.join(reddit_train_path, oracle_sent_fact_data_path)

reddit_valid_qa_path = os.path.join(reddit_valid_path, pro_qa_data_path)
reddit_valid_facts_path = os.path.join(reddit_valid_path, pro_facts_data_path)
reddit_valid_conv_path = os.path.join(reddit_valid_path, pro_conv_data_path)
reddit_valid_sent_fact_path = os.path.join(reddit_valid_path, sent_fact_data_path)
reddit_valid_oracle_sent_fact_path = os.path.join(reddit_valid_path, oracle_sent_fact_data_path)

reddit_test_qa_path = os.path.join(reddit_test_path, pro_qa_data_path)
reddit_test_facts_path = os.path.join(reddit_test_path, pro_facts_data_path)
reddit_test_conv_path = os.path.join(reddit_test_path, pro_conv_data_path)
reddit_test_sent_fact_path = os.path.join(reddit_test_path, sent_fact_data_path)
reddit_test_oracle_sent_fact_path = os.path.join(reddit_test_path, oracle_sent_fact_data_path)


