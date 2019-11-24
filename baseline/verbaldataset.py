"""VerbalDataset"""
import re
import json
from sklearn.model_selection import train_test_split
from torchtext.data import Field, Example, Dataset

from constants import (
    ANSWER_TOKEN, ENTITY_TOKEN, SOS_TOKEN, EOS_TOKEN,
    SRC_NAME, TRG_NAME, TRAIN_PATH, TEST_PATH
)
class VerbalDataset(object):
    """VerbalDataset class"""
    TOKENIZE_SEQ = lambda self, x: x.replace("?", " ?").\
                                     replace(".", " .").\
                                     replace(",", " ,").\
                                     replace("'", " '").\
                                     split()
    ANSWER_REGEX = r'\[.*?\]'
    def __init__(self, root_path):
        self.train_path = str(root_path) + TRAIN_PATH
        self.test_path = str(root_path) + TEST_PATH
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.src_field = None
        self.trg_field = None
        self.earl_entities = self._read_earl_entites(str(root_path) + '/baseline/earl_entities.json')

    def _read_earl_entites(self, path):
        entities = []
        with open(path) as json_file:
            entities = json.load(json_file)
        keys = [ent['uid'] for ent in entities]
        entities = dict(zip(keys, entities))
        return entities

    def _cover_answers(self, text):
        """
        Cover answers on text using an answer token
        """
        return re.sub(self.ANSWER_REGEX, ANSWER_TOKEN, text)

    def _cover_entities(self, uid, question, answer):
        """
        Cover entities on a given text.
        Since we use external entity recognizer it might
        miss some entities or cover wrong ones.
        A better approach will be to annotate all data
        and cover all entities. This should improve model
        performance.
        """
        # Try EARL for covering entities
        # EARL results are serialized
        data = self.earl_entities[uid]
        question_entities = data['question_entities']
        answer_entities = data['answer_entities']
        # we cover all recognized entitries by the same token
        # this has to be improved based on the number of entities.
        # For example if we have 2 entities we create 2 tokens e.g. <ent1> <ent2>
        # In this way we know the position of each entity in the translated output
        for ent in question_entities: question = question.replace(ent, ENTITY_TOKEN)
        for ent in answer_entities: answer = answer.replace(ent, ENTITY_TOKEN)

        return question, answer

    def _extract_question_answer(self, train, val, test):
        return [[data['question'], data['verbalized_answer']] for data in train], \
                [[data['question'], data['verbalized_answer']] for data in val], \
                [[data['question'], data['verbalized_answer']] for data in test]

    def _make_torchtext_dataset(self, data, fields):
        examples = [Example.fromlist(i, fields) for i in data]
        return Dataset(examples, fields)

    def load_data_and_fields(self, cover_entities=True):
        """
        Load verbalization data
        Create source and target fields
        """
        train, test, val = [], [], []
        # read data
        with open(self.train_path) as json_file:
            train = json.load(json_file)

        with open(self.test_path) as json_file:
            test = json.load(json_file)

        # cover answers
        for data in train: data.update((k, self._cover_answers(v)) for k, v in data.items() if k == "verbalized_answer")
        for data in test: data.update((k, self._cover_answers(v)) for k, v in data.items() if k == "verbalized_answer")

        # cover entities
        if cover_entities:
            for data in [train, test]:
                for example in data:
                    uid = example['uid']
                    question = example['question']
                    answer = example['verbalized_answer']
                    question, answer = self._cover_entities(uid, question, answer)
                    example.update(question=question, verbalized_answer=answer)

        # split test data to val-test
        test, val = train_test_split(test, test_size=0.5, shuffle=False)

        # extract question-answer pairs
        train, val, test = self._extract_question_answer(train, val, test)

        # create fields
        self.src_field = Field(tokenize=self.TOKENIZE_SEQ,
                               init_token=SOS_TOKEN,
                               eos_token=EOS_TOKEN,
                               lower=True,
                               include_lengths=True,
                               batch_first=True)
        self.trg_field = Field(tokenize=self.TOKENIZE_SEQ,
                               init_token=SOS_TOKEN,
                               eos_token=EOS_TOKEN,
                               lower=True,
                               batch_first=True)

        fields_tuple = [(SRC_NAME, self.src_field), (TRG_NAME, self.trg_field)]

        # create toechtext datasets
        self.train_data = self._make_torchtext_dataset(train, fields_tuple)
        self.valid_data = self._make_torchtext_dataset(val, fields_tuple)
        self.test_data = self._make_torchtext_dataset(test, fields_tuple)

        # build vocabularies
        self.src_field.build_vocab(self.train_data, min_freq=2)
        self.trg_field.build_vocab(self.train_data, min_freq=2)

    def get_data(self):
        """Return train, validation and test data objects"""
        return self.train_data, self.valid_data, self.test_data

    def get_fields(self):
        """Return source and target field objects"""
        return self.src_field, self.trg_field

    def get_vocabs(self):
        """Return source and target vocabularies"""
        return self.src_field.vocab, self.trg_field.vocab
