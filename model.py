import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    '''
    params:
        input_dim: vocab size of source language
        emb_dim: embedding dimension
        enc_hid_dim: hidden dimension of encoder
        dec_hid_dim: hidden dimension of decoder
        dropout: dropout rate
        n_layers: number of layers of GRU, default 2
    '''
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, n_layers: int=2):
        super().__init__()
        # convert input to embedding
        self.embedding = nn.Embedding(input_dim, emb_dim)

        '''
        input of GRU:
            input: tensor, shape (seq_len, batch, input_size)
            hidden: tensor, shape (n_layers * num_directions, batch, hidden_size)
        
        output of GRU:
            output: tensor, shape (seq_len, batch, num_directions * hidden_size)
            hidden: tensor, shape (n_layers * num_directions, batch, hidden_size)
        '''
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hid_dim, dec_hid_dim)

    
    '''
    params:
        src: tensor, shape (src_len, batch_size)
        src_len: tensor, shape (batch_size)
    '''
    def forward(self, src, src_len):
        # embedded (src_len, batch_size, emb_dim)
        embedded = self.dropout(self.embedding(src))
        # Maybe Sort First, but the doc says set to False, the PyTorch will sort it automatically
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len, enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed_embedded)
        # outputs: (src_len, batch_size, hidden_dim * num_directions)
        # hidden: (n_layers * num_directions, batch_size, hidden_dim)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        
        hidden = torch.tanh(self.fc(hidden))
        hidden = hidden.squeeze(0)
        # We only need the last hidden state
        # hidden = hidden[-1]
        # But decoder needs both of them, so we return the whole hidden
        # print(f'Hidden shape: {hidden.shape}')
        '''
        output: tensor, shape (src_len, batch_size, hidden_dim * num_directions)
        hidden: tensor, shape (n_layers, batch_size, hidden_dim)
        '''
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
    
    '''
    param:
        hidden: tensor, shape (batch_size, dec_hidden_dim)
        enc_outputs: tensor, shape (src_len, batch_size, enc_hid_dim)
        mask: point out the <pad> token, which is 0
    '''
    def forward(self, hidden, enc_outputs, mask):
        # Reshape enc_outputs to (batch_size, src_len, enc_hid_dim)
        enc_outputs = enc_outputs.permute(1, 0, 2)
        
        attention = torch.bmm(hidden.unsqueeze(1), enc_outputs.transpose(1, 2)).squeeze(1)
        # -1e10 is a very small number, so that the softmax will be close to 0
        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    '''
    params:
        output_dim: vocab size of target language
        emb_dim: embedding dimension
        enc_hid_dim: hidden dimension of encoder
        dec_hid_dim: hidden dimension of decoder
        dropout: dropout rate
        attention: attention layer
        n_layers: number of layers of GRU, default 2
    '''
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, attention, dropout, n_layers: int=2):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # Because we are using attention, the input size is emb_dim + enc_hid_dim
        self.rnn = nn.GRU(enc_hid_dim + emb_dim, dec_hid_dim, num_layers=n_layers)
        self.fc_out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    
    '''
    params:
        input: tensor, shape (batch_size)
        hidden: tensor, shape (n_layers_dec, batch_size, dec_hid_dim)
        enc_outputs: tensor, shape (src_len, batch_size, enc_hid_dim)
        mask: point out the <pad> token, which is 0
    '''
    def forward(self, input, hidden, enc_outputs, mask):
        # input shape (1, batch_size)
        input = input.unsqueeze(0)
        # embedded shape (1, batch_size, emb_dim)
        embedded = self.dropout(self.embedding(input))
        # atten_val shape (batch_size, src_len)
        atten_val = self.attention(hidden[-1], enc_outputs, mask)
        # atten_val shape (batch_size, 1, src_len)
        atten_val = atten_val.unsqueeze(1)
        # print(f'atten_val shape: {atten_val.shape}')
        # Shape (batch_size, src_len, enc_hid_dim)
        enc_outputs = enc_outputs.permute(1, 0, 2)
        # print('eo', enc_outputs.shape)
        weighted = torch.bmm(atten_val, enc_outputs)
        # weighted shape (batch_size, 1, enc_hid_dim)
        # print('weight', weighted.shape)
        weighted = weighted.permute(1, 0, 2)
        # Make the input of GRU to be (1, batch_size, enc_hid_dim + emb_dim)
        rnn_input = torch.cat((weighted, embedded), dim=2)
        # print(hidden.shape)
        output, hidden = self.rnn(rnn_input, hidden)

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        # prediction shape (batch_size, output_dim)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden.squeeze(0), atten_val.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device

    '''
    param: 
        src: tensor, shape (src_len, batch_size), a padded sequence
    '''
    def create_mask(self, src):
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask

    '''
    params:
        src: tensor, shape (src_len, batch_size)
        src_len: tensor, shape (batch_size)
        target: tensor, shape (target_len, batch_size)
        teacher_forcing_ratio: float, default 0.5
    '''
    def forward(self, src, src_len, target, teacher_forcing_ratio = 0.5):
        batch_size = src.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src, src_len)

        # <SOS>
        input = target[0, :]
        # Later, use the mask to make <PAD> become useless
        mask = self.create_mask(src)
        # Word by word
        for t in range(1, target_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[t] = output
            # TEACHING_FORCE or not
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            input = target[t] if teacher_force else top1

        return outputs

    def translate(self, src, src_len, target_dict, max_len=50):
        with torch.no_grad():
            enc_out, hidden = self.encoder(src, src_len)
        mask = self.create_mask(src)
        # print(f'Mask: {mask}')
        target_token = [target_dict.word_to_idx('<SOS>')]
        attentions = torch.zeros(max_len, 1, src.shape[0]).to(self.device)
        for t in range(max_len):
            target_ = torch.LongTensor([target_token[-1]]).to(self.device)
            with torch.no_grad():
                output, hidden, attention = self.decoder(target_, hidden, enc_out, mask)
            attentions[t] = attention
            
            pred_token = output.argmax(1).item()
            # print(f'Predicted token: {pred_token}')
            # print(f'Length of target token: {len(target_token)}')
            target_token.append(pred_token)
            
            # <EOS>
            if pred_token == target_dict.word_to_idx('<EOS>'):
                # Remove <SOS> and <EOS>
                target_token = target_token[1:-1]
                return target_token
        # Only remove <SOS>, as the sentence is not finished
        return target_token[1:]