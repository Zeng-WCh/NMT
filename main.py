from pathlib import Path
from util import get_data, WordDict


DATA_PREFIX = Path('./data')
TRAIN = DATA_PREFIX / 'train'
TEST = DATA_PREFIX / 'test'
VALID = DATA_PREFIX / 'validation'

SRC = 'src'
TARGET = 'target'
EXT = 'jsonl'
FILE = '{}.{}'

# Set max length of sentence
MAX_LENGTH = 32

def train(src: list, target: list):
    src_words = WordDict(False)
    target_words = WordDict(True)
    assert len(src) == len(target), 'Length of source and target should be the same.'
    for i in range(len(src)):
        src_words.add_sentence(src[i])
        target_words.add_sentence(target[i])
    


def main():
    train_src = get_data(TRAIN / FILE.format(SRC, EXT), MAX_LENGTH)
    train_target = get_data(TRAIN / FILE.format(TARGET, EXT), MAX_LENGTH)
    test_src = get_data(TEST / FILE.format(SRC, EXT), MAX_LENGTH)
    test_target = get_data(TEST / FILE.format(TARGET, EXT), MAX_LENGTH)
    valid_src = get_data(VALID / FILE.format(SRC, EXT), MAX_LENGTH)
    valid_target = get_data(VALID / FILE.format(TARGET, EXT), MAX_LENGTH)
    
    
if __name__ == '__main__':
    main()
