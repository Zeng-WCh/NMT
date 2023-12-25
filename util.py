import hanlp
import json
import tqdm
import re

import torch

from zhon.hanzi import punctuation as zh_punc
from string import punctuation as en_punc

from torch.utils.data import Dataset

tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

class WordDict:
    def __init__(self, is_chinese: bool = False):
        # Start of Sentence and End of Sentence
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        # To count the frequency of each word
        self.wordCount = dict()
        self.total_words = 4
        self.is_chinese = is_chinese
        # print(self.word2idx)
        # print(self.idx2word)
        
        
    def add_sentence(self, sentence: str):
        words = tokenize(sentence, self.is_chinese)
        for word in words:
            if self.wordCount.get(word) is not None:
                self.wordCount[word] += 1
            else:
                self.word2idx[word] = self.total_words
                self.idx2word[self.total_words] = word
                self.wordCount[word] = 1
                self.total_words += 1
                
                
    def tensorize(self, sentence: str, max_length: int = -1) -> torch.Tensor:
        words = tokenize(sentence, self.is_chinese)
        if max_length != -1 and len(words) > max_length:
            words = words[:max_length]
        # Padding
        if max_length != -1 and len(words) < max_length:
            words += ['<pad>'] * (max_length - len(words))
        # Add <sos> and <eos>
        words = ['<sos>'] + words + ['<eos>']
        # Convert to tensor
        return torch.tensor([self.word2idx[word] if self.word2idx.get(word) is not None else self.word2idx['<unk>'] for word in words])
    
    def __len__(self):
        return self.total_words


class SentenceDataSets(Dataset):
    def __init__(self, srcList, tarList, srcDict, targetDict):
        self.srcList = srcList
        self.tarList = tarList
        self.srcDict = srcDict
        self.targetDict = targetDict
    
    def __len__(self):
        return len(self.srcList)
    
    def __getitem__(self, idx):
        return self.srcDict.tensorize(self.srcList[idx]), self.targetDict.tensorize(self.tarList[idx])


def prepare_data(src: str, target: str, max_length: int = -1) -> tuple:
    src_data = get_data(src, max_length)
    target_data = get_data(target, max_length)
    assert len(src_data) == len(target_data), 'Length of source and target should be the same.'
    
    src_dict = WordDict(is_chinese=False)
    target_dict = WordDict(is_chinese=True)
    for i in tqdm.tqdm(range(len(src_data)), desc='Preparing data'):
        src_dict.add_sentence(src_data[i])
        target_dict.add_sentence(target_data[i])
    
    return src_data, target_data, src_dict, target_dict


def get_data_set(srcPath: str, tarPath: str, max_length: int = -1) -> SentenceDataSets:
    src_data, target_data, src_dict, target_dict = prepare_data(srcPath, tarPath, max_length)
    return SentenceDataSets(src_data, target_data, src_dict, target_dict)


def get_data(filename: str, max_length = -1) -> list:
    data = list()
    with open(filename, 'r') as f:
        for line in f:
            s = json.loads(line)['text'].strip()
            if max_length == -1:
                data.append(s)
            else:
                s = s[:max_length]
                data.append(s)
    return data


def tokenize(sentence: str, is_chinese: bool) -> list:
    if is_chinese:
        # Remove white space inside
        sentence = re.sub(r'\s+', '', sentence)
        # Remove punctuation like "《", "》", "，", "。", "？", "！", "：", "；", "、", "（", "）"
        rsub = re.sub(r'[{}]+'.format(zh_punc), '', sentence)
        return tok(rsub)
    else:
        # Remove punctuation and replace with white space
        sentence = re.sub(r'[{}]+'.format(en_punc), ' ', sentence)
        # Split by white space
        words = sentence.split()
        for i in range(len(words)):
            # Turn to lower case and remove white space
            words[i] = words[i].lower().strip()
        return words
    

