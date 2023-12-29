import argparse
import json
import torch
import pickle

from pathlib import Path
from tqdm import tqdm
from torchtext.data.metrics import bleu_score

from utils import prepare_data, WordDict
from model import Encoder, Decoder, Attention, Seq2Seq

DATA_PREFIX = Path('./data')
TRAIN = DATA_PREFIX / 'train'
TEST = DATA_PREFIX / 'test'
VALID = DATA_PREFIX / 'validation'

SRC = 'src'
TARGET = 'target'
EXT = 'jsonl'
# Use like FILEFMT.format([SRC|TARGET], EXT)
FILEFMT = '{}.{}'

TRAIN_ITER = 1000
EARLY_STOP = 5
BATCH_SIZE = 64
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 256
DEC_HID_DIM = 256
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
MAX_LENGTH = 50
LR = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_mask(src, padding):
    return (src != padding).permute(1, 0)

def translate_sentence(sentence: str, src_dict: WordDict, target_dict: WordDict, model: Seq2Seq, max_len=50) -> str:
    model.eval()
    sentence_vec = src_dict.to_tensor(sentence)
    # print(f'Sentence: {sentence_vec}')
    input_tensor = torch.LongTensor(sentence_vec).unsqueeze(1).to(device)
    input_len = torch.LongTensor([len(input_tensor)])
    
    res = model.translate(input_tensor, input_len, target_dict, max_len=max_len)
    # print(f'Result: {res}')
    # Also tokens
    tokens = [target_dict.idx_to_word(i) for i in res]
    
    return target_dict.to_sentence(res), tokens

def eval(model_path, param_path, src_dict_path, target_dict_path):
    _, _, _, _, sentence = prepare_data(TEST / FILEFMT.format(SRC, EXT), TEST / FILEFMT.format(TARGET, EXT), loaddir='./test')
    
    with open(param_path, 'r') as f:
        params = json.load(f)
    
    with open(src_dict_path, 'rb') as f:
        src_dict = pickle.load(f)
    with open(target_dict_path, 'rb') as f:
        target_dict = pickle.load(f)
    # sentence.reset_dict(src_dict, target_dict)
    
    enc = Encoder(params['ENC_VOC'], params['ENC_EMB_DIM'], params['ENC_HID_DIM'], params['DEC_HID_DIM'], params['ENC_DROPOUT'])
    attn = Attention()
    dec = Decoder(params['DEC_VOC'], params['DEC_EMB_DIM'], params['ENC_HID_DIM'], params['DEC_HID_DIM'], attn, params['DEC_DROPOUT'])
    model = Seq2Seq(enc, dec, src_dict.word_to_idx('<PAD>'), device)
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
    model = model.to(device)
    # result, result_tokens =  translate_sentence('Thank you!', src_dict, target_dict, model)
    # print(f'Result: {result}')
    # print(f'Tokens: {result_tokens}')
    batch = sentence.create_a_batches(1)
    
    pred_target = list()
    actual_target = list()
    
    results = list()
    
    
    for src, src_l, target, target_l in tqdm(batch, desc='Test Batch'):
        # print('Source: ', src_dict.to_sentence(src[0]))
        # print('Target: ', target_dict.to_sentence(target[0]))
        model.eval()
        with torch.no_grad():
            # print(len(src))
            # print(src[0].shape)
            output_s, tokens = translate_sentence(src_dict.to_sentence(src[0]), src_dict, target_dict, model, MAX_LENGTH)
            tar_tokens = [target_dict.idx_to_word(i) for i in target[0].tolist()]
            pred_target.append(tokens)
            actual_target.append([tar_tokens])
            current_bleu = bleu_score([tokens], [[tar_tokens]])
            current_result = {
                'src': src_dict.to_sentence(src[0][1:-1]),
                'target': target_dict.to_sentence(target[0][1:-1]),
                'model_output': output_s,
                'BLEU-4': current_bleu
            }
            # For debug usage
            if current_bleu != 0:
                print(f'Src: {current_result["src"]}')
                print(f'Result: {current_result["model_output"]}')
                print(f'Target: {current_result["target"]}')
                print(f'BLEU-4: {current_result["BLEU-4"]}')
                # print(f'Src: {current_result["src"]}, Result: {current_result["model_output"]}, Target: {current_result["target"]}, BLEU-4: {current_result["BLEU-4"]}')
            results.append(current_result)
    
    bleu = bleu_score(pred_target, actual_target)
    print(f'Total Bleu: {bleu}')
    
    with open('result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def generate_data():
    _, _, src_dict, target_dict, train_sens = prepare_data(TRAIN / FILEFMT.format(SRC, EXT), TRAIN / FILEFMT.format(TARGET, EXT), savedir='./train')
    valid_list, valid_tar_list, _, _, valid_sens = prepare_data(TEST / FILEFMT.format(SRC, EXT), TEST / FILEFMT.format(TARGET, EXT), savedir='./test')
    test_list, test_tar_list, _, _, test_sens = prepare_data(VALID / FILEFMT.format(SRC, EXT), VALID / FILEFMT.format(TARGET, EXT), savedir='./valid')
    print('Building dict, this may take a while...')
    for i in range(len(valid_list)):
        src_dict.add_a_sentence(valid_list[i])
        target_dict.add_a_sentence(valid_tar_list[i])
    
    for i in range(len(test_list)):
        src_dict.add_a_sentence(test_list[i])
        target_dict.add_a_sentence(test_tar_list[i])
    
    print('Preparing data...')
    
    train_sens.reset_dict(src_dict, target_dict)
    valid_sens.reset_dict(src_dict, target_dict)
    test_sens.reset_dict(src_dict, target_dict)
    
    print('Done.')
    
    with open('./train/dataset.pkl', 'wb') as f:
        pickle.dump(train_sens, f)
    
    with open('./test/dataset.pkl', 'wb') as f:
        pickle.dump(test_sens, f)
    
    with open('./valid/dataset.pkl', 'wb') as f:
        pickle.dump(valid_sens, f)
        
    with open('./src_dict.pkl', 'wb') as f:
        pickle.dump(src_dict, f)
    
    with open('./target_dict.pkl', 'wb') as f:
        pickle.dump(target_dict, f)
    

def train(model, param_path, src_dict_path, target_dict_path, reload=False):
    _, _, _, _, sentence = prepare_data(TRAIN / FILEFMT.format(SRC, EXT), TRAIN / FILEFMT.format(TARGET, EXT), loaddir='./train')
    _, _, _, _, sentence_valid = prepare_data(VALID / FILEFMT.format(SRC, EXT), VALID / FILEFMT.format(TARGET, EXT), loaddir='./valid')
    # Save the dict
    if not reload:
        valid_list, valid_tar_list, _, _, sentence_valid = prepare_data(VALID / FILEFMT.format(SRC, EXT), VALID / FILEFMT.format(TARGET, EXT), loaddir='./valid')
        test_list, test_tar_list, _, _, _ = prepare_data(TEST / FILEFMT.format(SRC, EXT), TEST / FILEFMT.format(TARGET, EXT), loaddir='./test')
        for i in range(len(valid_list)):
            src_dict.add_a_sentence(valid_list[i])
            target_dict.add_a_sentence(valid_tar_list[i])
        for i in range(len(test_list)):
            src_dict.add_a_sentence(test_list[i])
            target_dict.add_a_sentence(test_tar_list[i])
        sentence.reset_dict(src_dict, target_dict)
        sentence_valid.reset_dict(src_dict, target_dict)
    else:
        with open(src_dict_path, 'rb') as f:
            src_dict = pickle.load(f)
        with open(target_dict_path, 'rb') as f:
            target_dict = pickle.load(f)
    
    enc = Encoder(len(src_dict), ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    attn = Attention()
    dec = Decoder(len(target_dict), DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, attn, DEC_DROPOUT)
    
    model = Seq2Seq(enc, dec, src_dict.word_to_idx('<PAD>'), device).to(device)
    
    model_params = {
        'ENC_VOC': len(src_dict),
        'DEC_VOC': len(target_dict),
        'ENC_EMB_DIM': ENC_EMB_DIM,
        'DEC_EMB_DIM': DEC_EMB_DIM,
        'ENC_HID_DIM': ENC_HID_DIM,
        'DEC_HID_DIM': DEC_HID_DIM,
        'ENC_DROPOUT': ENC_DROPOUT,
        'DEC_DROPOUT': DEC_DROPOUT,
    }
    
    with open(param_path, 'w') as f:
        json.dump(model_params, f, indent=4)
    
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    
    PADDED_TOKENS = target_dict.word_to_idx('<PAD>')
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PADDED_TOKENS)
    
    best_loss = float('inf')
    best_epoch = 0
    
    for i in tqdm(range(TRAIN_ITER), desc='Training'):
        epoch_loss = 0
        batch = sentence.create_a_batches(BATCH_SIZE)
        for src, src_l, target, target_l in tqdm(batch, desc='Train Batch'):
            model.train()
            src = torch.from_numpy(src).to(device).long()
            src_l = torch.from_numpy(src_l).long()#.to(device)
            target = torch.from_numpy(target).to(device).long()
            target_l = torch.from_numpy(target_l).to(device).long()
            
            src = src.transpose(1, 0)
            target = target.transpose(1, 0)
            
            output = model(src, src_l, target)
            
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            
            target = target[1:].reshape(-1)
            # print(target.shape)
            # print(output.shape)
            loss = loss_fn(output, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            
            epoch_loss = epoch_loss + loss.item()
            torch.cuda.empty_cache()
        print(f'Epoch: {i}, Loss: {epoch_loss / len(batch)}')
        with open('training_loss.log', 'a') as f:
            f.write(f'Epoch: {i}, Loss: {epoch_loss / len(batch)}\n')
        
        # And validation
        # src_valid, tar_valid, _, _, sentence_valid = prepare_data(VALID / FILEFMT.format(SRC, EXT), VALID / FILEFMT.format(TARGET, EXT), loaddir='./valid')
        # # Replace the word dict 
        # sentence_valid.reset_dict(src_dict, target_dict)
        
        batch_valid = sentence_valid.create_a_batches(BATCH_SIZE)
        
        val_epoch_loss = 0
        for src, src_l, target, target_l in tqdm(batch_valid, desc='Valid Batch'):
            # model.eval()
            src = torch.from_numpy(src).to(device).long()
            src_l = torch.from_numpy(src_l).long()#.to(device)
            target = torch.from_numpy(target).to(device).long()
            target_l = torch.from_numpy(target_l).to(device).long()
            
            src = src.transpose(1, 0)
            target = target.transpose(1, 0)
            
            output = model(src, src_l, target)
            
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            
            target = target[1:].reshape(-1)
            # print(target.shape)
            # print(output.shape)
            
            loss = loss_fn(output, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            
            val_epoch_loss = val_epoch_loss + loss.item()
            torch.cuda.empty_cache()
        print(f'Valid Loss: {val_epoch_loss / len(batch_valid)}')
        with open('valid_loss.log', 'a') as f:
            f.write(f'Epoch: {i}, Loss: {val_epoch_loss / len(batch_valid)}\n')
            
            
        # Save model
        if epoch_loss < best_loss:
            torch.save(model.state_dict(), './model.pt')
            best_loss = epoch_loss
            best_epoch = 0
        else:
            best_epoch += 1
        # Early stopping
        if best_epoch == EARLY_STOP:
            print(f'Epoch: {i}, Loss: {epoch_loss / len(batch)}')
            print(f'Eearly stopping at epoch {i}')
            break
    

if __name__ == '__main__':
    # src, tar, _, _, _ = prepare_data(TEST / FILEFMT.format(SRC, EXT), TEST / FILEFMT.format(TARGET, EXT), loaddir='./train')
    
    # print(src[100])
    # print(tar[100])
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--generate_data', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--reload', action='store_true')
    
    parser.add_argument('--model', type=str, default='./model.pt')
    parser.add_argument('--src_dict', type=str, default='./src_dict.pkl')
    parser.add_argument('--target_dict', type=str, default='./target_dict.pkl')
    parser.add_argument('--params', type=str, default='./params.json')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--plot_loss', action='store_true')
    parser.add_argument('--loss', type=str, default='./training_loss.log')
    parser.add_argument('--output_loss', type=str, default='loss.png')
    
    args = parser.parse_args()
    
    if args.interactive:
        with open(args.src_dict, 'rb') as f:
            src_dict = pickle.load(f)
        with open(args.target_dict, 'rb') as f:
            target_dict = pickle.load(f)
        with open(args.params, 'r') as f:
            params = json.load(f)
        enc = Encoder(params['ENC_VOC'], params['ENC_EMB_DIM'], params['ENC_HID_DIM'], params['DEC_HID_DIM'], params['ENC_DROPOUT'])
        attn = Attention()
        dec = Decoder(params['DEC_VOC'], params['DEC_EMB_DIM'], params['ENC_HID_DIM'], params['DEC_HID_DIM'], attn, params['DEC_DROPOUT'])
        model = Seq2Seq(enc, dec, src_dict.word_to_idx('<PAD>'), device)
        with open(args.model, 'rb') as f:
            model.load_state_dict(torch.load(f))
        model = model.to(device)
        while True:
            try:
                s = input('>>> ')
            except EOFError:
                break
            # print(s)
            res, _ = translate_sentence(s, src_dict, target_dict, model, MAX_LENGTH)
            print(f'<<< {res}')
        exit()
    
    if args.plot_loss:
        pass
    
    if args.generate_data:
        generate_data()
    
    if args.train:
        train(args.model, args.params, args.src_dict, args.target_dict, args.reload)
        
    elif args.eval:
        eval(args.model, args.params, args.src_dict, args.target_dict)
    
    