import os
import json
import re
import argparse
from ltp_Label import ltp_Label

posmap = {"a": 0,  # adjective 形容词
          "b": 1,  # other noun-modifier 其他的修饰名词
          "c": 2,  # conjunction 连词
          "d": 3,  # adverb 副词
          "e": 4,  # exclamation 感叹词
          "g": 5,  # morpheme 语素
          "h": 6,  # prefix 前缀
          "i": 7,  # idiom 成语
          "j": 8,  # abbreviation 缩写
          "k": 9,  # suffix 后缀
          "m": 10,  # number 数字
          "n": 11,  # general noun 一般名词
          "nd": 12,  # direction noun 方向名词
          "nh": 13,  # person name 人名
          "ni": 14,  # organization name 机构名
          "nl": 15,  # location noun 地点名
          "ns": 16,  # geographical name 地理名
          "nt": 17,  # temporal noun 时间名
          "nz": 18,  # other proper noun 其他名词
          "o": 19,  # onomatopoeia 拟声词
          "p": 20,  # preposition 介词
          "q": 21,  # quantity 量词
          "r": 22,  # pronoun 代词
          "u": 23,  # auxiliary 助词
          "v": 24,  # verb 动词
          "wp": 25,  # punctuation 标点符号
          "ws": 26,  # foreign words 国外词
          "x": 27,  # non-lexeme 不构成词
          "z": 28,  # descriptive words
          "%": 29}
dpmap = {"SBV": 0,  # subject-verb 主谓关系
         "VOB": 1,  # verb-object 动宾关系(直接宾语)
         "IOB": 2,  # indirect-object 间宾关系(间接宾语)
         "FOB": 3,  # fronting-object 前置宾语
         "DBL": 4,  # double 兼语
         "ATT": 5,  # attribute 定中关系
         "ADV": 6,  # adverbial 状中结构
         "CMP": 7,  # complement 动补结构
         "COO": 8,  # coordinate 并列关系
         "POB": 9,  # preposition-object 介宾关系
         "LAD": 10,  # left adjunct 左附加关系
         "RAD": 11,  # right adjunct 右附加关系
         "IS": 12,  # independent structure 独立结构
         "HED": 13,  # head 核心关系
         "WP": 14}  # punctuation 标点


def gettag(words, postags, arcs):
    poss = []
    dps = []
    head = []
    headmap = {}
    i = 0
    for fetch in zip(words, postags, arcs):
        word, pos, arc = fetch
        for w in word:
            poss.append(posmap[pos])
            dps.append(dpmap[arc.relation])
            headmap[arc.head] = i
    for fetch in zip(words, postags, arcs):
        word, pos, arc = fetch
        for w in word:
            head.append(headmap[arc.head])

    return poss, dps, head


def _argparse():
    parser = argparse.ArgumentParser(description="Add POS and DP tags")
    parser.add_argument('-ltp',  action='store', dest='ltp_data_dir', required=True,
                        default="ltp_data_v3.4.0", help='The path of source data set')
    parser.add_argument('-rf', action='store', dest='rf',
                        default="", help='The path of source data set')
    parser.add_argument('-wf', action='store', dest='wf',
                        default="", help='The path of output data set')
    return parser.parse_args()


if __name__ == "__main__":
    arg = _argparse()

    rf = open(arg.rf, 'r', encoding='utf-8')
    wf = open(arg.wf, 'w', encoding='utf-8')
    labeler = ltp_Label(arg.ltp_data_dir)
    lines = rf.readlines()
    line_num = len(lines)

    for i, line in enumerate(lines):
        sentence = json.loads(line)
        natural = re.sub(' ', '', sentence['sentence'])
        words, postags, arcs = labeler.get_features(natural)
        poss, dp, head = gettag(words, postags, arcs)
        sentence['sentence'] = natural
        sentence['pos_tag'] = poss
        sentence['dp_tag'] = dp
        sentence['head_rag'] = head
        s = json.dumps(sentence, ensure_ascii=False) + '\n'
        wf.write(s)
