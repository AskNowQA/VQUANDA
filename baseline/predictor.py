"""Predictor"""
import torch
from constants import (
    SOS_TOKEN, EOS_TOKEN, RNN_NAME
)

class Predictor(object):
    """Predictor class"""
    def __init__(self, model, src_vocab, trg_vocab, device):
        self.model = model
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.device = device

    def _predict_cnn_step(self, tokens):
        self.model.eval()
        tokenized_sentence = [SOS_TOKEN] + [t.lower() for t in tokens] + [EOS_TOKEN]
        numericalized = [self.src_vocab.stoi[token] for token in tokenized_sentence]
        src_tensor = torch.LongTensor(numericalized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            encoder_out = self.model.encoder(src_tensor)

        outputs = [self.trg_vocab.stoi[SOS_TOKEN]]

        for _ in range(self.model.decoder.max_positions):
            trg_tensor = torch.LongTensor(outputs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output, attention = self.model.decoder(trg_tensor, encoder_out, src_tokens=src_tensor)

            prediction = output.argmax(2)[:, -1].item()

            outputs.append(prediction)

            if prediction == self.trg_vocab.stoi[EOS_TOKEN]:
                break

        translation = [self.trg_vocab.itos[i] for i in outputs]

        return translation[1:], attention

    def _predict_rnn_step(self, tokens):
        self.model.eval()
        with torch.no_grad():
            tokenized_sentence = [SOS_TOKEN] + [t.lower() for t in tokens] + [EOS_TOKEN]
            numericalized = [self.src_vocab.stoi[t] for t in tokenized_sentence]

            src_len = torch.LongTensor([len(numericalized)]).to(self.device)
            tensor = torch.LongTensor(numericalized).unsqueeze(1).to(self.device)

            translation_tensor_logits, attention = self.model(tensor.t(), src_len, None)

            translation_tensor = torch.argmax(translation_tensor_logits.squeeze(1), 1)
            translation = [self.trg_vocab.itos[t] for t in translation_tensor]

        return translation[1:], attention

    def predict(self, tokens):
        """Perform prediction on given tokens"""
        return self._predict_rnn_step(tokens) if self.model.name == RNN_NAME else \
                self._predict_cnn_step(tokens)
