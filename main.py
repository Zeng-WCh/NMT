import torch
import tqdm

from torch.utils.data import DataLoader
from pathlib import Path

from util import get_data_set
from model import Seq2Seq

DATA_PREFIX = Path('./data')
TRAIN = DATA_PREFIX / 'train'
TEST = DATA_PREFIX / 'test'
VALID = DATA_PREFIX / 'validation'

SRC = 'src'
TARGET = 'target'
EXT = 'jsonl'
# Use like FILEFMT.format([SRC|TARGET], EXT)
FILEFMT = '{}.{}'

# Set max length of sentence
MAX_LENGTH = 512

TRAIN_ITER = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    # Load Data
    train_data_loader = DataLoader(get_data_set(TRAIN / FILEFMT.format(SRC, EXT), TRAIN / FILEFMT.format(TARGET, EXT), MAX_LENGTH), batch_size=1, shuffle=True)
    model = Seq2Seq()
    
    loss_function = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for i in tqdm.tqdm(range(TRAIN_ITER)):
        for src, target in train_data_loader:
            src = src.to(device)
            target = target.to(device)
            
            output = model(src, target)
            loss = loss_function(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Initialize model
    # nmt_model = Seq2Seq(Encoder(100, 100, 2, 0.1), Decoder(100, 100, 2, 0.1), device).to(device)
    
    
    

def main():
    train()

    
if __name__ == '__main__':
    main()
