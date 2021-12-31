# -*- coding: utf-8 -*-
import os
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import Parser


class ltp_Label(object):
    def __init__(self, ltp_data_dir):
        self.cws_model_path = os.path.join(ltp_data_dir, 'cws.model')
        self.pos_model_path = os.path.join(ltp_data_dir, 'pos.model')
        self.par_model_path = os.path.join(ltp_data_dir, 'parser.model')
        self.segmentor = Segmentor()
        self.postagger = Postagger()
        self.parser = Parser()
        self.segmentor.load(self.cws_model_path)
        self.postagger.load(self.pos_model_path)
        self.parser.load(self.par_model_path)

    def __del__(self):
        self.segmentor.release()
        self.postagger.release()
        self.parser.release()

    def get_features(self, sentence):
        words = self.segmentor.segment(sentence)
        postags = self.postagger.postag(words)
        arcs = self.parser.parse(words, postags)
        words = list(words)
        postags = list(postags)
        arcs = list(arcs)
        return words, postags, arcs
