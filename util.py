import json
import hanlp
import re
import copy

from zhon.hanzi import punctuation as zh_punc
from string import punctuation as en_punc

tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

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
    
class WordDict:
    def __init__(self, is_chinese: bool):
        # Start of Sentence and End of Sentence
        self.word2idx = {'<sos>': 0, '<eos>': 1}
        self.idx2word = {0: '<sos>', 1: '<eos>'}
        # To count the frequency of each word
        self.wordCount = dict()
        self.total_words = 2
        self.is_chinese = is_chinese
        
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
