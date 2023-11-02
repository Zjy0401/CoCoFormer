import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
import random
import pickle
from utilities.constants import *
from utilities.device import get_device
from tqdm import tqdm

from .positional_encoding import PositionalEncoding
from .rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR, TransformerEncoderLayerRPR_, \
    TransformerEncoderPastLayer, TransformerEncoderLayer, TransformerEncoder

from typing import Dict, Iterable, Callable
from torch.nn.init import *
from utilities.argument_funcs import parse_train_args, print_train_args, write_model_params

# MusicTransformer
class CoCoformer(nn.Module):

    def __init__(self, word2event, event2word, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
                 dropout=0.1, max_sequence=2048, c_max_seq=256, b_max_seq=1024, rpr=False):
        super(CoCoformer, self).__init__()

        self.dummy = DummyDecoder()
        self.nlayers = n_layers
        self.nhead = num_heads
        self.d_model = d_model
        self.d_ff = dim_feedforward
        self.dropout = dropout
        self.max_seq = max_sequence
        self.c_max_seq = c_max_seq
        self.b_max_seq = b_max_seq
        self.rpr = rpr
        # word2event and event2word:
        self.word2event = word2event
        self.event2word = event2word

        # past layer of chord
        self.cpast_layer_dmodel = d_model
        self.cpast_layer_nhead = 8
        self.cpast_dim_forward = 256
        self.cpast_layer_max_seq = 256
        self.cpast_layer_nlayers = 1

        # past layer of beats
        self.bpast_layer_dmodel = d_model
        self.bpast_layer_nhead = 8
        self.bpast_dim_forward = 256
        self.bpast_layer_max_seq = 1024
        self.bpast_layer_nlayers = 1

        # Input embedding
        self.n_embedding = nn.Embedding(VOCAB_SIZE, self.d_model)
        self.c_embedding = nn.Embedding(VOCAB_SIZE, self.cpast_layer_dmodel)
        self.b_embedding = nn.Embedding(VOCAB_SIZE, self.bpast_layer_dmodel)
        # Positional encoding
        self.n_positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq)
        self.c_positional_encoding = PositionalEncoding(self.cpast_layer_dmodel, self.dropout, self.cpast_layer_max_seq)
        self.b_positional_encoding = PositionalEncoding(self.bpast_layer_dmodel, self.dropout, self.bpast_layer_max_seq)

        # Base transformer
        if not self.rpr:
            # To make a decoder-only transformer we need to use masked encoder layers
            # Dummy decoder to essentially just return the encoder output
            encoder_norm = LayerNorm(self.d_model)
            encoder_past_layer = TransformerEncoderPastLayer(self.cpast_layer_dmodel, self.cpast_layer_nhead,
                                                             self.cpast_dim_forward, self.bpast_layer_dmodel,
                                                             self.bpast_layer_nhead, self.bpast_dim_forward,
                                                             self.d_model, self.nhead,
                                                             self.d_ff, self.dropout)
            encoder_layer = TransformerEncoderLayer(self.d_model, self.nhead, self.d_ff, self.dropout)
            encoder = TransformerEncoder(encoder_layer, self.nlayers, encoder_past_layer, self.max_seq, self.c_max_seq,
                                         self.b_max_seq, encoder_norm)
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=0, dropout=self.dropout,  # activation=self.ff_activ,
                dim_feedforward=self.d_ff, custom_encoder=encoder, custom_decoder=self.dummy
            )
        # RPR Transformer
        elif self.rpr:
            encoder_norm = LayerNorm(self.d_model)
            encoder_layer = TransformerEncoderLayerRPR(self.d_model, self.nhead, self.d_ff, self.dropout,
                                                       er_len=self.max_seq)
            encoder_past_layer = TransformerEncoderLayerRPR_(self.cpast_layer_dmodel, self.cpast_layer_nhead,
                                                             self.cpast_dim_forward, self.bpast_layer_dmodel,
                                                             self.bpast_layer_nhead, self.bpast_dim_forward,
                                                             self.d_model, self.nhead,
                                                             self.d_ff, self.dropout, er_len=self.max_seq)
            encoder = TransformerEncoderRPR(encoder_layer, self.nlayers, encoder_past_layer, self.max_seq,
                                            self.c_max_seq, self.b_max_seq, encoder_norm)

            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=0, dropout=self.dropout,  # activation=self.ff_activ,
                dim_feedforward=self.d_ff, custom_decoder=self.dummy, custom_encoder=encoder
            )

        # Final output is a softmaxed linear layer
        # TODO: verify the size of linear
        self.Norm1 = nn.LayerNorm(1024)
        self.ReLU = nn.ReLU()
        self.Norm2 = nn.LayerNorm(181)
        self.Dropout = nn.Dropout(dropout)
        self.transLinear = nn.Linear(256, 256)
        self.Wout1 = nn.Linear(self.d_model, 1024)
        self.Wout2 = nn.Linear(1024, 1024)
        self.Wout3 = nn.Linear(1024, VOCAB_SIZE)
        self.softmax = nn.Softmax(dim=-1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    # forward
    def forward(self, x1, x2, x3, mask=True):

        args = parse_train_args()
        # for pure-Transformer:
        # Transformer module:
        if mask is True:
            if args.gpu[0] != -1:
                mask = self.transformer.generate_square_subsequent_mask(x1.shape[1]).cuda(device=args.gpu[0])
            else:
                mask = self.transformer.generate_square_subsequent_mask(x1.shape[1]).cpu()
        else:
            mask = None
        # Input shape is (max_seq, batch_size, d_model)
        x_n = self.n_embedding(x1)
        x_n = x_n.permute(1, 0, 2)
        x_n = self.n_positional_encoding(x_n)

        x_c = self.c_embedding(x2)
        x_c = x_c.permute(1, 0, 2)
        x_c = self.c_positional_encoding(x_c)

        x_b = self.b_embedding(x3)
        x_b = x_b.permute(1, 0, 2)
        x_b = self.b_positional_encoding(x_b)

        # Since there are no true decoder layers, the tgt is unused
        # Pytorch wants src and tgt to have some equal dims however
        x_out = self.transformer(src=torch.cat((x_n, x_c, x_b), dim=0), tgt=x_n,
                                 src_mask=mask)
        # x_out = self.transformer(src=x_transformer, tgt=x_transformer, src_mask=mask)
        # Back to (batch_size, max_seq, d_model)
        x_out = x_out.permute(1, 0, 2)

        # concat
        # x_concat = torch.cat([x_out, x_out2], dim=1)
        y = self.Dropout(self.Norm1(self.ReLU(self.Wout1(x_out))))
        y = self.Dropout(self.Norm1(self.ReLU(self.Wout2(y))))
        y = self.Wout3(y)
        # y = self.Wout2(y)
        # y = self.softmax(y)

        del mask

        # They are trained to predict the next note in sequence (we don't need the last one)
        return y

    # unconditional generate
    def generate(self, primer=None, target_seq_length=1024, beam=0, beam_chance=1.0):

        assert (not self.training), "Cannot generate while in training mode"

        print("Generating sequence of max length:", target_seq_length)

        gen_seq = torch.full((1, target_seq_length), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())

        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(get_device())

        # print("primer:",primer)
        # print(gen_seq)
        cur_i = num_primer
        while cur_i < target_seq_length:
            # gen_seq_batch     = gen_seq.clone()
            y = self.softmax(self.forward(gen_seq[..., :cur_i]))[..., :len(self.word2event)]
            token_probs = y[:, cur_i - 1, :]

            if beam == 0:
                beam_ran = 2.0
            else:
                beam_ran = random.uniform(0, 1)

            if beam_ran <= beam_chance:
                token_probs = token_probs.flatten()
                top_res, top_i = torch.topk(token_probs, beam)

                beam_rows = top_i // VOCAB_SIZE
                beam_cols = top_i % VOCAB_SIZE

                gen_seq = gen_seq[beam_rows, :]
                gen_seq[..., cur_i] = beam_cols

            else:
                distrib = torch.distributions.categorical.Categorical(probs=token_probs)
                next_token = distrib.sample()
                # print("next token:",next_token)
                gen_seq[:, cur_i] = next_token

                # Let the transformer decide to end if it wants to
                # if next_token == TOKEN_END:
                #     print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                #     break

            cur_i += 1
            if cur_i % 50 == 0:
                print(cur_i, "/", target_seq_length)

        return gen_seq[:, :cur_i]

    # conditional generate
    def conditional_generate(self, beats, chord, seq, c, bs, ba, bt, bb, target_seq_length=1024, beam=0, beam_chance=1.0):

        assert (not self.training), "Cannot generate while in training mode"
        print("Generating sequence of max length:", target_seq_length)
        chord = torch.tensor(chord, device=get_device()).unsqueeze(0)
        beats = torch.tensor(beats, device=get_device()).unsqueeze(0)

        gen_seq = torch.full((1, target_seq_length), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        primer = torch.tensor([c[0], bs[0], seq[0], ba[0]])
        primer_num = 1  # decide key to add
        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(get_device())

        # print("primer:",primer)
        # print(gen_seq)
        cur_i = num_primer
        # first input: C B N B
        cur_i_n = 1
        cur_i_b = 2
        cur_i_c = 1
        check_error = 0
        pbar = tqdm(total=len(seq)*9)
        while cur_i < target_seq_length:
            a = gen_seq[..., :cur_i].cpu().numpy()
            # gen_seq_batch = gen_seq.clone()
            # print("input:", gen_seq[..., :cur_i], chord[..., :cur_i_c], beats[..., :cur_i_b])
            y = self.softmax(self.forward(gen_seq[..., :cur_i], chord[..., :cur_i_c],
                                          beats[..., :cur_i_b]))[..., :len(self.word2event)]
            token_probs = y[:, cur_i - 1, :]
            # check for y
            distrib = torch.distributions.categorical.Categorical(probs=token_probs)
            next_token = distrib.sample()
            if check_error > 256:
                print("error! regenerate!")
                return False
            # next token is the next token
            if cur_i % 9 == 1:  # token is chord, next token must be beats
                if not 178 < next_token < 191:  # if it is not beat
                    check_error += 1
                    continue
            if cur_i % 9 in [2, 4, 6, 8]:  # this token must be beat, next token must be note
                if not next_token < 129:  # if it is not note
                    check_error += 1
                    continue
            else:  # this token must be note, next token must be chord or beat
                if not 128 < next_token < 191:  # if it is chord or beat
                    check_error += 1
                    continue

            if beam == 0:
                beam_ran = 2.0
            else:
                beam_ran = random.uniform(0, 1)

            if beam_ran <= beam_chance:
                token_probs = token_probs.flatten()
                top_res, top_i = torch.topk(token_probs, beam)

                beam_rows = top_i // VOCAB_SIZE
                beam_cols = top_i % VOCAB_SIZE

                gen_seq = gen_seq[beam_rows, :]
                gen_seq[..., cur_i] = beam_cols

            else:
                # print("next token:",next_token)
                gen_seq[:, cur_i] = next_token
                cur_i += 1
                pbar.update(1)
                cur_i_n += 1
                if cur_i % 9 == 0 and primer_num < len(seq):
                    # add C B_S N_S B_A
                    gen_seq[:, cur_i] = chord.squeeze()[primer_num]
                    gen_seq[:, cur_i+1] = torch.tensor(bs[primer_num], device=get_device())
                    gen_seq[:, cur_i+2] = torch.tensor(seq[primer_num], device=get_device())
                    gen_seq[:, cur_i+3] = torch.tensor(ba[primer_num], device=get_device())
                    primer_num += 1
                    cur_i += 4
                    pbar.update(4)
                    cur_i_n += 1
                    cur_i_b += 2
                    cur_i_c += 1
                    # a = gen_seq[..., :cur_i].cpu().numpy()
                if cur_i % 9 != 0 and cur_i % 9 != 4 and primer_num < len(seq) + 1:
                    # add B
                    gen_seq[:, cur_i] = beats.squeeze()[cur_i_b]
                    cur_i_b += 1
                    cur_i_n += 1
                    cur_i += 1
                    pbar.update(1)
                    # a = gen_seq[..., :cur_i].cpu().numpy()
                if primer_num == len(seq) and cur_i == len(seq) * 9:
                    print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                    break
            # print(cur_i, "/", target_seq_length)

        print("all errors:%d" % check_error)
        return gen_seq[:, :cur_i]

# Used as a dummy to nn.Transformer
# DummyDecoder
class DummyDecoder(nn.Module):

    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask):

        return memory


class Discriminator(nn.Module):
    """
    to judge the true sample or fake
    return fake or true
    """
    def __init__(self, input_emb=1, d_model=256, nhead=4, d_ff=512, dropout=0.5, out_emb=1024):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(input_emb, d_model)
        self.transformer = TransformerEncoderLayer(d_model, nhead, d_ff, dropout)
        self.linear2 = nn.Linear(d_model, out_emb)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.Norm1 = nn.LayerNorm(d_model)
        self.Norm2 = nn.LayerNorm(out_emb)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, x, labels):
        x = x.float().unsqueeze(2)
        x = self.dropout(self.Norm1(self.linear1(x)))
        x = self.transformer(x)
        logits = self.dropout(self.Norm2(self.linear2(x)))
        logits = self.sigmoid(self.relu(self.maxpool(logits)))
        logits = logits.reshape(logits.shape[0] * logits.shape[1], -1)
        labels = labels.reshape(logits.shape[0] * logits.shape[1], -1)
        loss = self.loss(logits, labels)

        # import numpy as np
        # logits = logits.cpu().detach().numpy()
        # labels = labels.cpu().detach().numpy()
        # loss = []
        # for i in logits:
        #     loss.append(np.log(1-1/(1+np.exp(i[0]))))
        output = (loss, logits)

        return output

    def _reset_parameters(self):

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class PureTransformer(nn.Module):

    def __init__(self, word2event, event2word, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
                 dropout=0.1, max_sequence=2048, c_max_seq=256, b_max_seq=1024, rpr=False):
        super(PureTransformer, self).__init__()
        self.dummy = DummyDecoder()
        self.nlayers = n_layers
        self.nhead = num_heads
        self.d_model = d_model
        self.d_ff = dim_feedforward
        self.dropout = dropout
        self.max_seq = max_sequence
        self.rpr = rpr
        # word2event and event2word:
        self.word2event = word2event
        self.event2word = event2word
        # Input embedding
        self.embedding = nn.Embedding(VOCAB_SIZE, self.d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq)

        self.transformer = nn.Transformer(
            d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
            num_decoder_layers=0, dropout=self.dropout,  # activation=self.ff_activ,
            dim_feedforward=self.d_ff, custom_decoder=self.dummy
        )

        # Final output is a softmaxed linear layer
        self.Wout = nn.Linear(self.d_model, VOCAB_SIZE)
        self.softmax = nn.Softmax(dim=-1)

    # forward
    def forward(self, x, mask=True):

        if mask is True:
            mask = self.transformer.generate_square_subsequent_mask(x[0].shape[1]).to(get_device())
        else:
            mask = None

        x = self.embedding(x)

        # Input shape is (max_seq, batch_size, d_model)
        x = x.permute(1, 0, 2)

        x = self.positional_encoding(x)

        # Since there are no true decoder layers, the tgt is unused
        # Pytorch wants src and tgt to have some equal dims however
        x_out = self.transformer(src=x, tgt=x, src_mask=mask)

        # Back to (batch_size, max_seq, d_model)
        x_out = x_out.permute(1, 0, 2)

        y = self.Wout(x_out)
        # y = self.softmax(y)

        del mask

        # They are trained to predict the next note in sequence (we don't need the last one)
        return y
