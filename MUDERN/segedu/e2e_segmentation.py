import re
from argparse import ArgumentParser
from nltk.tokenize import word_tokenize
import pickle
import numpy as np
import torch
from solver import TrainSolver
from model import PointerNetworks
import json
import nltk.data
from tqdm import tqdm


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"RE_DIGITS":1,"UNKNOWN":2,"PADDING":0}
        self.word2count = {"RE_DIGITS":1,"UNKNOWN":1,"PADDING":1}
        self.index2word = {0: "PADDING", 1: "RE_DIGITS", 2: "UNKNOWN"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.strip('\n').strip('\r').split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def mytokenizer(inS,all_dict):
    #repDig = re.sub(r'\d+[\.,/]?\d+','RE_DIGITS',inS)
    repDig = re.sub(r'\d*[\d,]*\d+', 'RE_DIGITS', inS)
    toked = word_tokenize(repDig)
    or_toked = word_tokenize(inS)
    re_unk_list = []
    ori_list = []
    # Yifan: fix bug in difference of length
    if len(toked) != len(or_toked):
        toked = or_toked

    for (i,t) in enumerate(toked):
        if t not in all_dict and t not in ['RE_DIGITS']:
            re_unk_list.append('UNKNOWN')
            ori_list.append(or_toked[i])
        else:
            re_unk_list.append(t)
            ori_list.append(or_toked[i])

    labey_edus = [0]*len(re_unk_list)
    labey_edus[-1] = 1
    return ori_list,re_unk_list,labey_edus


def get_mapping(X,Y,D):
    X_map = []
    for w in X:
        if w in D:
            X_map.append(D[w])
        else:
            X_map.append(D['UNKNOWN'])

    X_map = np.array([X_map])
    Y_map = np.array([Y])
    return X_map,Y_map


def main_input_output(inputstring, args):
    # all_voc = r'segedu/all_vocabulary.pickle'
    all_voc = args.vocab
    model_dir = args.model
    voca = pickle.load(open(all_voc, 'rb'))
    voca_dict = voca.word2index

    ori_X, X, Y = mytokenizer(inputstring, voca_dict)

    X_in, Y_in = get_mapping(X, Y, voca_dict)

    mymodel = PointerNetworks(voca_size =2, voc_embeddings=np.ndarray(shape=(2,300), dtype=float),word_dim=300, hidden_dim=10,is_bi_encoder_rnn=True,rnn_type='GRU',rnn_layers=3,
                 dropout_prob=0.5,use_cuda=False,finedtuning=True,isbanor=True)

    mymodel = torch.load(model_dir, map_location=lambda storage, loc: storage)
    mymodel.use_cuda = False

    mymodel.eval()
    mysolver = TrainSolver(mymodel, '', '', '', '', '',
                           batch_size=1, eval_size=1, epoch=10, lr=1e-2, lr_decay_epoch=1, weight_decay=1e-4,
                           use_cuda=False)

    test_batch_ave_loss, test_pre, test_rec, test_f1, visdata = mysolver.check_accuracy(X_in, Y_in)

    start_b = visdata[3][0]
    end_b = visdata[2][0] + 1
    segments = []

    for i, END in enumerate(end_b):
        START = start_b[i]
        segments.append(' '.join(ori_X[START:END]))

    return segments


def parsing_snippet(snippet):
    title_matched = re.match(r'#.{2,}\n\n', snippet)
    if title_matched:
        title_span = title_matched.span()
        title = snippet[title_span[0]:title_span[1]].strip('\n').strip()
        context = snippet[title_span[1]:]
    else:
        title = ''
        context = snippet
    # check if exist bullets
    bullet_segmenter = '* '
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    is_bullet = False
    if bullet_segmenter in context:
        clauses = []
        is_bullet = True
        bullet_position = [m.start() for m in re.finditer('\* ', context)]
        if bullet_position[0] != 0:
            prompt = context[:bullet_position[0]].strip()
            if '. ' in prompt:
                prompt_sentences = sent_detector.tokenize(prompt)
                clauses.extend(prompt_sentences)
            else:
                clauses.append(prompt)
        for idx in range(len(bullet_position)):
            current_start_pos = bullet_position[idx]
            next_start_pos = bullet_position[idx + 1] if idx + 1 < len(bullet_position) else len(context) + 1
            clauses.append(context[current_start_pos:next_start_pos].strip('\n').strip())
    else:
        clauses = sent_detector.tokenize(context)
    return (title, clauses, is_bullet)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('fin', default='',
                        help='input data file')
    parser.add_argument('dout', default='',
                        help='directory to store output files')
    parser.add_argument('--vocab', default='',
                        help='vocabulary.pickle')
    parser.add_argument('--model', default='',
                        help='pt')

    args = parser.parse_args()

    with open(args.fin) as f:
        data = json.load(f)
    for ex in tqdm(data):
        title, clauses, is_bullet = parsing_snippet(ex['snippet'])
        ex['parsed'] = {}
        ex['parsed']['title'] = title
        ex['parsed']['clauses'] = clauses
        ex['parsed']['is_bullet'] = is_bullet
        if is_bullet:
            ex['parsed']['has_edu'] = False
        else:
            ex['parsed']['has_edu'] = True
            ex['parsed']['edus'] = []
            for rule_sentence in ex['parsed']['clauses']:
                rule_edus = main_input_output(rule_sentence, args)
                ex['parsed']['edus'].append(rule_edus)

    with open(args.dout, 'wt') as f:
        json.dump(data, f, indent=2)