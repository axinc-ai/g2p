# -*- coding: utf-8 -*-
# /usr/bin/python
'''
By kyubyong park(kbpark.linguist@gmail.com) and Jongseok Kim(https://github.com/ozmig77)
https://www.github.com/kyubyong/g2p
'''
from nltk import pos_tag
from nltk.corpus import cmudict
import nltk
from nltk.tokenize import TweetTokenizer
word_tokenize = TweetTokenizer().tokenize
import torch
import torch.nn as nn
import torch.nn.functional as F
import codecs
import re
import os
import unicodedata
from builtins import str as unicode
from expand import normalize_numbers
import numpy as np

try:
    nltk.data.find('taggers/averaged_perceptron_tagger.zip')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('corpora/cmudict.zip')
except LookupError:
    nltk.download('cmudict')

dirname = os.path.dirname(__file__)

def construct_homograph_dictionary():
    f = os.path.join(dirname,'homographs.en')
    homograph2features = dict()
    for line in codecs.open(f, 'r', 'utf8').read().splitlines():
        if line.startswith("#"): continue # comment
        headword, pron1, pron2, pos1 = line.strip().split("|")
        homograph2features[headword.lower()] = (pron1.split(), pron2.split(), pos1)
    return homograph2features

class G2pEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.load_variables()

    def load_variables(self):
        self.variables = np.load(os.path.join(dirname,'checkpoint20.npz'))
        self.enc_emb = torch.tensor(self.variables["enc_emb"])  # (29, 64). (len(graphemes), emb)
        self.enc_w_ih = torch.tensor(self.variables["enc_w_ih"])  # (3*128, 64)
        self.enc_w_hh = torch.tensor(self.variables["enc_w_hh"])  # (3*128, 128)
        self.enc_b_ih = torch.tensor(self.variables["enc_b_ih"])  # (3*128,)
        self.enc_b_hh = torch.tensor(self.variables["enc_b_hh"])  # (3*128,)

        self.dec_emb = torch.tensor(self.variables["dec_emb"])  # (74, 64). (len(phonemes), emb)
        self.dec_w_ih = torch.tensor(self.variables["dec_w_ih"])  # (3*128, 64)
        self.dec_w_hh = torch.tensor(self.variables["dec_w_hh"])  # (3*128, 128)
        self.dec_b_ih = torch.tensor(self.variables["dec_b_ih"])  # (3*128,)
        self.dec_b_hh = torch.tensor(self.variables["dec_b_hh"])  # (3*128,)
        self.fc_w = torch.tensor(self.variables["fc_w"])  # (74, 128)
        self.fc_b = torch.tensor(self.variables["fc_b"])  # (74,)

    def sigmoid(self, x):
        return torch.sigmoid(x)

    def grucell(self, x, h, w_ih, w_hh, b_ih, b_hh):
        rzn_ih = F.linear(x, w_ih, b_ih)
        rzn_hh = F.linear(h, w_hh, b_hh)

        rz_ih, n_ih = torch.split(rzn_ih, int(rzn_ih.shape[-1] * 2 / 3), dim=-1)
        rz_hh, n_hh = torch.split(rzn_hh, int(rzn_hh.shape[-1] * 2 / 3), dim=-1)

        rz = self.sigmoid(rz_ih + rz_hh)
        r, z = torch.split(rz, int(rz.shape[-1] / 2), dim=-1)

        n = torch.tanh(n_ih + r * n_hh)
        h = (1 - z) * n + z * h

        return h

    def gru(self, x, steps, w_ih, w_hh, b_ih, b_hh, h0=None):
        if h0 is None:
            h0 = torch.zeros((x.shape[0], w_hh.shape[1]), dtype=torch.float32)
        h = h0  # initial hidden state
        outputs = torch.zeros((x.shape[0], steps, w_hh.shape[1]), dtype=torch.float32)
        for t in range(steps):
            h = self.grucell(x[:, t, :], h, w_ih, w_hh, b_ih, b_hh)  # (b, h)
            outputs[:, t, :] = h
        return outputs

    def encode(self, x):
        x = F.embedding(x, self.enc_emb)
        return x
    
    def encode_and_fs_decode(self, x, len_words):
        # encoder
        enc = self.encode(x).unsqueeze(0)
        enc = self.gru(enc, len_words + 1, self.enc_w_ih, self.enc_w_hh,
                       self.enc_b_ih, self.enc_b_hh, h0=torch.zeros((1, self.enc_w_hh.shape[-1]), dtype=torch.float32))
        last_hidden = enc[:, -1, :]

        # decoder
        dec = F.embedding(torch.tensor([2]), self.dec_emb).unsqueeze(0)  # 2: <s>
        h = last_hidden
        return dec, h

    def forward(self, x, len_words):
        return self.encode_and_fs_decode(x, len_words)
    
class G2pDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.load_variables()

    def sigmoid(self, x):
        return torch.sigmoid(x)

    def grucell(self, x, h, w_ih, w_hh, b_ih, b_hh):
        rzn_ih = F.linear(x, w_ih, b_ih)
        rzn_hh = F.linear(h, w_hh, b_hh)

        rz_ih, n_ih = torch.split(rzn_ih, int(rzn_ih.shape[-1] * 2 / 3), dim=-1)
        rz_hh, n_hh = torch.split(rzn_hh, int(rzn_hh.shape[-1] * 2 / 3), dim=-1)

        rz = self.sigmoid(rz_ih + rz_hh)
        r, z = torch.split(rz, int(rz.shape[-1] / 2), dim=-1)

        n = torch.tanh(n_ih + r * n_hh)
        h = (1 - z) * n + z * h

        return h

    def gru(self, x, steps, w_ih, w_hh, b_ih, b_hh, h0=None):
        if h0 is None:
            h0 = torch.zeros((x.shape[0], w_hh.shape[1]), dtype=torch.float32)
        h = h0  # initial hidden state
        outputs = torch.zeros((x.shape[0], steps, w_hh.shape[1]), dtype=torch.float32)
        for t in range(steps):
            h = self.grucell(x[:, t, :], h, w_ih, w_hh, b_ih, b_hh)  # (b, h)
            outputs[:, t, :] = h
        return outputs

    def load_variables(self):
        self.variables = np.load(os.path.join(dirname,'checkpoint20.npz'))
        self.enc_emb = torch.tensor(self.variables["enc_emb"])  # (29, 64). (len(graphemes), emb)
        self.enc_w_ih = torch.tensor(self.variables["enc_w_ih"])  # (3*128, 64)
        self.enc_w_hh = torch.tensor(self.variables["enc_w_hh"])  # (3*128, 128)
        self.enc_b_ih = torch.tensor(self.variables["enc_b_ih"])  # (3*128,)
        self.enc_b_hh = torch.tensor(self.variables["enc_b_hh"])  # (3*128,)

        self.dec_emb = torch.tensor(self.variables["dec_emb"])  # (74, 64). (len(phonemes), emb)
        self.dec_w_ih = torch.tensor(self.variables["dec_w_ih"])  # (3*128, 64)
        self.dec_w_hh = torch.tensor(self.variables["dec_w_hh"])  # (3*128, 128)
        self.dec_b_ih = torch.tensor(self.variables["dec_b_ih"])  # (3*128,)
        self.dec_b_hh = torch.tensor(self.variables["dec_b_hh"])  # (3*128,)
        self.fc_w = torch.tensor(self.variables["fc_w"])  # (74, 128)
        self.fc_b = torch.tensor(self.variables["fc_b"])  # (74,)

    def decode(self, dec, h):
        h = self.grucell(dec.squeeze(1), h, self.dec_w_ih, self.dec_w_hh, self.dec_b_ih, self.dec_b_hh)  # (b, h)
        logits = F.linear(h, self.fc_w, self.fc_b)
        pred = logits.argmax().item()
        dec = F.embedding(torch.tensor([pred]), self.dec_emb).unsqueeze(0)
        return pred, dec, h

    def forward(self, dec, h):
        return self.decode(dec, h)


class G2p(object):
    def __init__(self):
        super().__init__()

        self.graphemes = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")
        self.phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                                             'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
                                                             'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                                             'EY2', 'F', 'G', 'HH',
                                                             'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L',
                                                             'M', 'N', 'NG', 'OW0', 'OW1',
                                                             'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH',
                                                             'UH0', 'UH1', 'UH2', 'UW',
                                                             'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
        self.g2idx = {g: idx for idx, g in enumerate(self.graphemes)}
        self.idx2g = {idx: g for idx, g in enumerate(self.graphemes)}

        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}

        self.cmu = cmudict.dict()
        self.homograph2features = construct_homograph_dictionary()

    def tokenize(self, word):
        chars = list(word) + ["</s>"]
        x = [self.g2idx.get(char, self.g2idx["<unk>"]) for char in chars]
        return torch.tensor(x)
        #return x

    def predict(self, word):
        encoder = G2pEncoder()
        decoder = G2pDecoder()

        x = self.tokenize(word)

        dec, h = encoder.encode_and_fs_decode(x, len(word))
        #print(x.shape)
        #print(dec.shape)
        #print(h.shape)

        preds = []
        for i in range(20):
            pred, dec, h = decoder.decode(dec, h)
            if pred == 3: break  # 3: </s>
            preds.append(pred)

        preds = [self.idx2p.get(idx, "<unk>") for idx in preds]
        return preds

    def __call__(self, text):

        # preprocessing
        text = unicode(text)
        text = normalize_numbers(text)
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                       if unicodedata.category(char) != 'Mn')  # Strip accents
        text = text.lower()
        text = re.sub("[^ a-z'.,?!\-]", "", text)
        text = text.replace("i.e.", "that is")
        text = text.replace("e.g.", "for example")

        # tokenization
        words = word_tokenize(text)
        tokens = pos_tag(words)  # tuples of (word, tag)

        # steps
        prons = []
        for word, pos in tokens:
            if re.search("[a-z]", word) is None:
                pron = [word]

            elif word in self.homograph2features:  # Check homograph
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                else:
                    pron = pron2
            elif word in self.cmu:  # lookup CMU dict
                pron = self.cmu[word][0]
            else: # predict for oov
                pron = self.predict(word)

            prons.extend(pron)
            prons.extend([" "])

        return prons[:-1]

if __name__ == '__main__':
    texts = ["I have $250 in my pocket.", # number -> spell-out
             "popular pets, e.g. cats and dogs", # e.g. -> for example
             "I refuse to collect the refuse around here.", # homograph
             "I'm an activationist."] # newly coined word
    g2p = G2p()
    for text in texts:
        out = g2p(text)
        print(out)

        # export
        #text = "activationist"
        #encoder = G2pEncoder()
        #x = g2p.tokenize(text)
        #len_words = len(text)
        #print(x)
        #print(len_words)
        #torch.onnx.export(encoder,
        #                (x, len_words),
        #                "encode_and_fs_decode.onnx",
        #                input_names=['x', 'len_words'],
        #                output_names=['dec', 'h'],
        #                dynamic_axes={'x': {0: 'seq_len'}})
