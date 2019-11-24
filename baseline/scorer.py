"""Scorer"""
import nltk

class BleuScorer(object):
    """Blue scorer class"""
    def __init__(self):
        self.results = []
        self.score = 0
        self.instances = 0

    def example_score(self, reference, hypothesis):
        """Calculate blue score for one example"""
        return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)

    def data_score(self, data, predictor):
        """Score complete list of data"""
        for example in data:
            reference = [t.lower() for t in example.trg]
            hypothesis, _ = predictor.predict(example.src)
            blue_score = self.example_score(reference, hypothesis)
            self.results.append({
                'reference': reference,
                'hypothesis': hypothesis,
                'blue_score': blue_score
            })
            self.score += blue_score
            self.instances += 1

        return self.score / self.instances

    def average_score(self):
        """Return bleu average score"""
        return self.score / self.instances

    def reset(self):
        """Reset object properties"""
        self.results = []
        self.score = 0
        self.instances = 0
