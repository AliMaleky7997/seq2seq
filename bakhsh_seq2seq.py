import json
import random
import re

import torch
import torch.nn as nn
from torch import optim

SOS_token = 0
EOS_token = 1


class PoemHeja:
    def __init__(self, name):
        if name == 'cv':
            self.name = 'cv'
            self.char2index = {'c': 2, 'v': 3}
            self.index2char = {0: 'SOS', 1: 'EOS', 2: 'c', 3: 'v'}
            self.n_chars = 4
        else:
            self.name = 'poem'
            self.char2index = {'ب': 2, 'ه': 3, ' ': 4, 'ن': 5, 'ا': 6, 'م': 7, 'خ': 8, 'د': 9, 'و': 10, 'ج': 11,
                               'ر': 12, 'ک': 13, 'ز': 14, 'ی': 15, 'ت': 16, 'ع': 17, 'ش': 18, 'گ': 19, 'س': 20, 'پ': 21,
                               'ف': 22, 'چ': 23, 'ل': 24, 'ژ': 25, 'ق': 26}
            self.index2char = {0: 'SOS', 1: 'EOS', 2: 'ب', 3: 'ه', 4: ' ', 5: 'ن', 6: 'ا', 7: 'م', 8: 'خ', 9: 'د',
                               10: 'و', 11: 'ج', 12: 'ر', 13: 'ک', 14: 'ز', 15: 'ی', 16: 'ت', 17: 'ع', 18: 'ش', 19: 'گ',
                               20: 'س', 21: 'پ', 22: 'ف', 23: 'چ', 24: 'ل', 25: 'ژ', 26: 'ق'}
            self.n_chars = len(self.index2char)


def normalizing(st: str):
    st = re.sub(r"[ذظض]", "ز", st)
    st = re.sub(r"[ثص]", "س", st)
    st = re.sub(r"غ", "ق", st)
    st = re.sub(r"ط", "ت", st)
    st = re.sub(r"ح", "ه", st)
    st = re.sub(r"ء", "ع", st)
    st = re.sub(r"(?<= )(ع)", "ا", st)
    st = re.sub(r"آ", "ا", st)
    st = re.sub(r"اا", "ا", st)
    st = re.sub(r"(?<= )(ا)", "عا", st)
    return st


def prepare_data():
    print("Reading lines...")

    pairs = []
    with open('bakhsh_data.json', 'r', encoding='utf-8') as file:
        for l in file.readlines():
            temp_line = json.loads(l)
            pairs.append([normalizing(temp_line[1]), temp_line[0]])

    input_char = PoemHeja('poem')
    output_char = PoemHeja('cv')

    return input_char, output_char, pairs

input_char, output_char, pairs = prepare_data()
print(random.choice(pairs))


# ============================== End of Data Preparing ==============================

# ============================== Start of Seq2Seq Modeling ==============================

# --------------- Encoding ---------------

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


import torch.nn.functional as F


# --------------- Decoding ---------------

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# ============================== Start of Data Preparing ==============================


# --------------- Attention Decoding ---------------

MAX_LENGTH = 60


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================== END of Seq2Seq Modeling ==============================

# ============================== Start of Seq2Seq Training ==============================

# --------------- Preparing Training Data ---------------

def indexes_from_sentence(poem_heja: PoemHeja, sentence):
    return [poem_heja.char2index[char] for char in list(sentence)]


def tensor_from_sentence(poem_heja, sentence):
    indexes = indexes_from_sentence(poem_heja, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(pair):
    input_tensor = tensor_from_sentence(input_char, pair[0])
    target_tensor = tensor_from_sentence(output_char, pair[1])
    return input_tensor, target_tensor


# --------------- Training the Model ---------------

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_iters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    print("Training...")
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensors_from_pair(random.choice(pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % ('Still training... it is now: ', iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


# ============================== Start of Seq2Seq Training ==============================

# ============================== Start of Seq2Seq Evaluation ==============================

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_char, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_char.index2char[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluate_randomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', make_heja(pair[1]))
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = make_heja(''.join(output_words))
        print('<', output_sentence)
        print('')


def make_heja(st: str):
    return re.sub(r"(?<=\w)(cv)", r" \1", st)


hidden_size = 100
encoder1 = EncoderRNN(input_char.n_chars, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_char.n_chars, dropout_p=0.1).to(device)

train_iters(encoder1, attn_decoder1, 500, print_every=250)
evaluate_randomly(encoder1, attn_decoder1, 3)
