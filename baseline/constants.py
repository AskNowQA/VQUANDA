"""Constants for the baseline models"""
SEED = 42
RNN_NAME = 'rnn'
CNN_NAME = 'cnn'

ATTENTION_1 = 'bahdanau'
ATTENTION_2 = 'luong'

GPU = 'gpu'
CPU = 'cpu'
CUDA = 'cuda'

TRAIN_PATH = '/dataset/train.json'
TEST_PATH = '/dataset/test.json'
CHECKPOINT_PATH = '/model/'

ANSWER_TOKEN = '<ans>'
ENTITY_TOKEN = '<ent>'
EOS_TOKEN = '<eos>'
SOS_TOKEN = '<sos>'
PAD_TOKEN = '<pad>'

SRC_NAME = 'src'
TRG_NAME = 'trg'
