import hanlp
import tqdm
import torch
import os
import pickle
import re
import sys
import json

import numpy as np

from string import punctuation as en_punc
from zhon.hanzi import punctuation as zh_punc
from pathlib import Path

tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

# Return a list of token and cleaned sentence
def tokenize(sentence: str, is_chinese: bool) -> tuple([list, str]):
    if is_chinese:
        # Remove white space inside
        sentence = re.sub(r'\s+', '', sentence)
        # Remove punctuation like "《", "》", "，", "。", "？", "！", "：", "；", "、", "（", "）"
        rsub = re.sub(r'[{}]+'.format(zh_punc), '', sentence)
        tokens = tok(rsub)
        return tokens, rsub
    else:
        # Remove punctuation and replace with white space
        sentence = re.sub(r'[{}]+'.format(en_punc), ' ', sentence)
        sentence = sentence.strip().lower()
        # Split by white space
        words = sentence.split()
        for i in range(len(words)):
            # Turn to lower case and remove white space
            words[i] = words[i].lower().strip()
        return words, sentence

# print(tokenize("《论语》一书由孔子子弟撰写。", True))

class WordDict:
    def __init__(self, is_chinese: bool=False):
        self.is_ch = is_chinese
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {k: v for v, k in self.word2idx.items()}
        self.words = {}
        self.num_words = 4
        
    def add_a_sentence(self, sentence: str) -> str:
        tokens, cleaned = tokenize(sentence, self.is_ch)
        
        for word in tokens:
            if self.words.get(word) is None:
                self.word2idx[word] = self.num_words
                self.idx2word[self.num_words] = word
                self.num_words += 1
                self.words[word] = 1
            else:
                self.words[word] += 1

        return cleaned
                
    def __len__(self):
        return self.num_words

    def to_sentence(self, vector) -> str:
        words = [self.idx2word[idx] for idx in vector]
        if self.is_ch:
            return ''.join(words)
        return ' '.join(words)
    
    def to_tensor(self, sentence: str) -> torch.Tensor:
        tokens, _ = tokenize(sentence, self.is_ch)
        tokens = ['<SOS>'] + tokens + ['<EOS>']
        # print(tokens)
        # print(len(tokens))
        return torch.LongTensor([self.word2idx.get(word, self.word2idx['<UNK>']) for word in tokens])
    
    def word_to_idx(self, word: str) -> int:
        return self.word2idx.get(word, self.word2idx['<UNK>'])

    def idx_to_word(self, idx: int) -> str:
        return self.idx2word.get(idx, 0)


class SentencesDataSet:
    def __init__(self, src_list, target_list, src_dict, target_dict):
        self.src_list = src_list
        self.target_list = target_list
        self.src_dict = src_dict
        self.target_dict = target_dict
        
        self.src_vec_list = list()
        self.target_vec_list = list()
        
    
    @staticmethod
    def padding_sentence(data):
        # Get the longgest sentence
        lengths = [len(s) for s in data]
        n_nums = len(data)
        # print(f'n_nums {n_nums}')
        max_length = np.max(lengths)
        
        output = np.zeros((n_nums, max_length), dtype=np.int32)
        actual_len = np.array(lengths, dtype=np.int32)
        
        # Padding to the longest, and rest fill with 0
        for i, sen in enumerate(data):
            output[i, :lengths[i]] = sen
            
        return output, actual_len
        
    
    def create_a_batches(self, batch_size):
        if len(self.src_vec_list) == 0:
            for i in range(len(self.src_list)):
                self.src_vec_list.append(self.src_dict.to_tensor(self.src_list[i]))
                self.target_vec_list.append(self.target_dict.to_tensor(self.target_list[i]))
        idx_list = np.arange(0, len(self.src_list), batch_size)
        np.random.shuffle(idx_list)
        
        batches = list()
        
        for idx in idx_list:
            batches.append(np.arange(idx, min(idx + batch_size, len(self.src_list))))
        
        batch = list()
        
        for i in batches:
            # print('i', len(i))
            src_list = [self.src_vec_list[j] for j in i]
            target_list = [self.target_vec_list[j] for j in i]
            
            src_list, src_acl = self.padding_sentence(src_list)
            target_list, tar_acl = self.padding_sentence(target_list)
            batch.append((src_list, src_acl, target_list, tar_acl))
            
        return batch
    
    def reset_dict(self, src_dict, target_dict):
        self.src_dict = src_dict
        self.target_dict = target_dict
        
        self.src_vec_list.clear()
        self.target_vec_list.clear()
        
        for i in range(len(self.src_list)):
            self.src_vec_list.append(self.src_dict.to_tensor(self.src_list[i]))
            self.target_vec_list.append(self.target_dict.to_tensor(self.target_list[i]))


def get_data(filename: str) -> list:
    data = list()
    with open(filename, 'r') as f:
        for line in f:
            s = json.loads(line)['text'].strip()
            data.append(s)
    return data

def prepare_data(src: str, target: str, savedir=None, loaddir=None) -> tuple:
    if loaddir is None:
        src_data = get_data(src)
        target_data = get_data(target)
        assert len(src_data) == len(target_data), 'Length of source and target should be the same.'
        
        src_dict = WordDict(is_chinese=False)
        target_dict = WordDict(is_chinese=True)
        for i in tqdm.tqdm(range(len(src_data)), desc='Preparing data'):
            src_data[i] = src_dict.add_a_sentence(src_data[i])
            target_data[i] = target_dict.add_a_sentence(target_data[i])
        
        sentence = SentencesDataSet(src_data, target_data, src_dict, target_dict)
        
        if savedir is not None:
            if not os.path.exists(savedir):
                os.mkdir(savedir)
            with open(Path(savedir) / 'src_dict.pkl', 'wb') as f:
                pickle.dump(src_dict, f)
            with open(Path(savedir) / 'target_dict.pkl', 'wb') as f:
                pickle.dump(target_dict, f)
            with open(Path(savedir) / 'src_data.pkl', 'wb') as f:
                pickle.dump(src_data, f)
            with open(Path(savedir) / 'target_data.pkl', 'wb') as f:
                pickle.dump(target_data, f)
            with open(Path(savedir) / 'dataset.pkl', 'wb') as f:
                pickle.dump(sentence, f)
        return src_data, target_data, src_dict, target_dict, sentence
    else:
        if not os.path.exists(loaddir):
            sys.stderr.write(f'No dir named \'{loaddir}\'\n')
            sys.exit(1)
        with open(Path(loaddir) / 'src_dict.pkl', 'rb') as f:
            src_dict = pickle.load(f)
        with open(Path(loaddir) / 'target_dict.pkl', 'rb') as f:
            target_dict = pickle.load(f)
        with open(Path(loaddir) / 'src_data.pkl', 'rb') as f:
            src_data = pickle.load(f)
        with open(Path(loaddir) / 'target_data.pkl', 'rb') as f:
            target_data = pickle.load(f)
        with open(Path(loaddir) / 'dataset.pkl', 'rb') as f:
            sentence = pickle.load(f)
        return src_data, target_data, src_dict, target_dict, sentence
    