import os
import pickle
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utilities.constants import *
from utilities.device import cpu_device

import numpy as np
import shutil
import mido

SEQUENCE_START = 0


# EPianoDataset
class MultiJSFDataset(Dataset):

    def __init__(self, root, max_seq=2048, random_seq=True):
        self.root = root
        self.max_seq = max_seq
        self.chord_max_seq = 256
        self.beat_max_seq = 1024
        self.random_seq = random_seq

        fs = [os.path.join(root, f) for f in os.listdir(self.root)]
        self.data_files = [f for f in fs if os.path.isfile(f)]

    # __len__
    def __len__(self):

        return len(self.data_files)

    # __getitem__
    def __getitem__(self, idx):
        # # All data on cpu to allow for the Dataloader to multithread
        # i_stream = open(self.data_files[idx], "rb")
        # # return pickle.load(i_stream), None
        # # raw_mid = torch.tensor(mido.MidiFile(self.data_files[idx]), dtype=TORCH_LABEL_TYPE, device=cpu_device())
        # # input SATB
        # raw_mid = torch.tensor(pickle.load(i_stream)[0], dtype=TORCH_LABEL_TYPE, device=cpu_device())
        #
        # x, tgt = process_midi(raw_mid, self.max_seq, self.random_seq)
        # # input chord
        # raw_chord = torch.tensor(pickle.load(i_stream)[1], dtype=TORCH_LABEL_TYPE, device=cpu_device())
        # chord = process_midi(raw_chord, self.max_seq, self.random_seq)
        # i_stream.close()

        with open(self.data_files[idx], 'rb') as f:
            data = pickle.load(f)

            # input of SATB
            raw_mid = torch.tensor(data[0], dtype=TORCH_LABEL_TYPE, device=cpu_device())
            raw, tgt = process_midi(raw_mid, self.max_seq, self.random_seq)

            # input of chord
            raw_chord = torch.tensor(data[1], dtype=TORCH_LABEL_TYPE, device=cpu_device())
            chord, c_ = process_midi(raw_chord, self.chord_max_seq, self.random_seq)

            # input of Beat
            raw_beat = torch.tensor(data[2], dtype=TORCH_LABEL_TYPE, device=cpu_device())
            beat, b_ = process_midi(raw_beat, self.beat_max_seq, self.random_seq)

            # for multi-loss:
            chord_pad = torch.zeros_like(chord) + TOKEN_PAD
            beat_pad = torch.zeros_like(beat) + TOKEN_PAD
            x = ((raw, chord, beat), (raw, chord_pad, beat_pad))
            tgt = ((tgt, c_, b_), (tgt, chord_pad, beat_pad))
        return x, tgt



class JSFDataset(Dataset):

    def __init__(self, root, max_seq=2048, random_seq=True):
        self.root = root
        self.max_seq = max_seq
        self.chord_max_seq = 256
        self.beat_max_seq = 1024
        self.random_seq = random_seq

        fs = [os.path.join(root, f) for f in os.listdir(self.root)]
        self.data_files = [f for f in fs if os.path.isfile(f)]

    # __len__
    def __len__(self):
        """
        ----------
        Author: Damon Gwinn
        ----------
        How many data files exist in the given directory
        ----------
        """

        return len(self.data_files)

    # __getitem__
    def __getitem__(self, idx):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Gets the indexed midi batch. Gets random sequence or from start depending on random_seq.

        Returns the input and the target.
        ----------
        """
        # # All data on cpu to allow for the Dataloader to multithread
        # i_stream = open(self.data_files[idx], "rb")
        # # return pickle.load(i_stream), None
        # # raw_mid = torch.tensor(mido.MidiFile(self.data_files[idx]), dtype=TORCH_LABEL_TYPE, device=cpu_device())
        # # input SATB
        # raw_mid = torch.tensor(pickle.load(i_stream)[0], dtype=TORCH_LABEL_TYPE, device=cpu_device())
        #
        # x, tgt = process_midi(raw_mid, self.max_seq, self.random_seq)
        # # input chord
        # raw_chord = torch.tensor(pickle.load(i_stream)[1], dtype=TORCH_LABEL_TYPE, device=cpu_device())
        # chord = process_midi(raw_chord, self.max_seq, self.random_seq)
        # i_stream.close()

        with open(self.data_files[idx], 'rb') as f:
            data = pickle.load(f)

            # input of SATB
            raw_mid = torch.tensor(data[0], dtype=TORCH_LABEL_TYPE, device=cpu_device())
            raw, tgt = process_midi(raw_mid, self.max_seq, self.random_seq)

            # input of chord
            raw_chord = torch.tensor(data[1], dtype=TORCH_LABEL_TYPE, device=cpu_device())
            chord, c_ = process_midi(raw_chord, self.chord_max_seq, self.random_seq)

            # input of Beat
            raw_beat = torch.tensor(data[2], dtype=TORCH_LABEL_TYPE, device=cpu_device())
            beat, b_ = process_midi(raw_beat, self.beat_max_seq, self.random_seq)

            x = (raw, chord, beat)
            tgt = (tgt, c_, b_)
        return x, tgt


# process_midi
def process_midi(raw_mid, max_seq, random_seq):

    x = torch.full((max_seq,), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=cpu_device())
    tgt = torch.full((max_seq,), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=cpu_device())

    raw_len = len(raw_mid)
    full_seq = max_seq + 1  # Performing seq2seq

    if raw_len == 0:
        return x, tgt

    if raw_len < full_seq:
        x[:raw_len] = raw_mid
        tgt[:raw_len - 1] = raw_mid[1:]
        tgt[raw_len - 1] = TOKEN_END
    else:
        # Randomly selecting a range
        if random_seq:
            end_range = raw_len - full_seq
            start = random.randint(SEQUENCE_START, end_range)

        # Always taking from the start to as far as we can
        else:
            start = SEQUENCE_START

        end = start + full_seq

        data = raw_mid[start:end]

        x = data[:max_seq]
        tgt = data[1:full_seq]

    # print("x:",x)
    # print("tgt:",tgt)

    return x, tgt


# create_JSF_datasets
def create_jsf_datasets(dataset_root, max_seq, random_seq=True):

    train_root = os.path.join(dataset_root, "train")
    # val_root = os.path.join(dataset_root, "val")
    test_root = os.path.join(dataset_root, "test")

    train_dataset = MultiJSFDataset(train_root, max_seq, random_seq)
    # val_dataset = JSFDataset(val_root, max_seq, random_seq)
    test_dataset = MultiJSFDataset(test_root, max_seq, random_seq)

    return train_dataset, test_dataset


# compute_epiano_accuracy
def compute_jsf_accuracy(out, tgt):

    softmax = nn.Softmax(dim=-1)
    out = torch.argmax(softmax(out), dim=-1)

    # test for bug:
    # out = np.array(out.cpu())
    # tgt = np.array(tgt.cpu())
    # only calculate note:

    # out = out[:, :2048].flatten()
    # tgt = tgt[:, :2048].flatten()

    out = out.flatten()
    tgt = tgt.flatten()

    mask = (tgt != TOKEN_PAD)

    out = out[mask]
    tgt = tgt[mask]

    # Empty
    if (len(tgt) == 0):
        return 1.0

    num_right = (out == tgt)
    num_right = torch.sum(num_right).type(TORCH_FLOAT)

    acc = num_right / len(tgt)

    return acc


# compute_epiano_accuracy
def compute_jsf_ter(out, tgt):

    softmax = nn.Softmax(dim=-1)
    out = torch.argmax(softmax(out), dim=-1)

    out = out.flatten()
    tgt = tgt[0][0]
    tgt = tgt.flatten()
    mask = (tgt != TOKEN_PAD)

    out = out[mask]
    tgt = tgt[mask]
    # Empty
    if (len(tgt) == 0):
        return 1.0
    # TER:
    C_t, B_St, N_St, B_At, N_At, B_Tt, N_Tt, B_Bt, N_Bt = 0, 0, 0, 0, 0, 0, 0, 0, 0
    C_f, B_Sf, N_Sf, B_Af, N_Af, B_Tf, N_Tf, B_Bf, N_Bf = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(len(out)):
        if i % 9 == 0:
            if out[i] == tgt[i]:
                C_t += 1
            else:
                C_f += 1

        if i % 9 == 1:
            if out[i] == tgt[i]:
                B_St += 1
            else:
                B_Sf += 1
        if i % 9 == 2:
            if out[i] == tgt[i]:
                N_St += 1
            else:
                N_Sf += 1

        if i % 9 == 3:
            if out[i] == tgt[i]:
                B_At += 1
            else:
                B_Af += 1
        if i % 9 == 4:
            if out[i] == tgt[i]:
                N_At += 1
            else:
                N_Af += 1

        if i % 9 == 5:
            if out[i] == tgt[i]:
                B_Tt += 1
            else:
                B_Tf += 1
        if i % 9 == 6:
            if out[i] == tgt[i]:
                N_Tt += 1
            else:
                N_Tf += 1

        if i % 9 == 7:
            if out[i] == tgt[i]:
                B_Bt += 1
            else:
                B_Bf += 1
        if i % 9 == 8:
            if out[i] == tgt[i]:
                N_Bt += 1
            else:
                N_Bf += 1

    C = C_t / (C_t + C_f)

    BS = B_St / (B_St + B_Sf)
    NS = N_St / (N_St + N_Sf)

    BA = B_At / (B_At + B_Af)
    NA = N_At / (N_At + N_Af)

    BT = B_Tt / (B_Tt + B_Tf)
    NT = N_Tt / (N_Tt + N_Tf)

    BB = B_Bt / (B_Bt + B_Bf)
    NB = N_Bt / (N_Bt + N_Bf)

    return [C, NS, BS, NA, BA, NT, BT, NB, BB]


def to_onehot(chord):
    '''
	create one-hot vector from the pitch class of the note.
	'''
    onehot = []
    for i in range(12):
        if i == chord:
            onehot.append(1)
        else:
            onehot.append(0)
    return np.array(onehot)


def calculate_vocal_part_entropy(x, y):
    softmax = nn.Softmax(dim=-1)
    y = torch.argmax(softmax(y), dim=-1)
    mask = (x != TOKEN_PAD)
    y = y[mask]
    x = x[mask]
    y = y.cpu().numpy()
    x = x.cpu().numpy()

    S_y, A_y, T_y, B_y = yseq2SATB(y)
    S_x, A_x, T_x, B_x = xseq2SATB(x)

    p_sx, p_ax, p_tx, p_bx = prob_distri(S_x), prob_distri(A_x), prob_distri(T_x), prob_distri(B_x)
    p_sy, p_ay, p_ty, p_by = prob_distri(S_y), prob_distri(A_y), prob_distri(T_y), prob_distri(B_y)

    return vocal_part_entropy(p_sx, p_sy) + vocal_part_entropy(p_ax, p_ay) + \
        vocal_part_entropy(p_tx, p_ty) + vocal_part_entropy(p_bx, p_by)

def vocal_part_entropy(x, y):
    '''
    input: seqs x and y distribution
    output: entropy
    '''
    return entropy(x) - entropy(y)

def calculate_vocal_part_cross_entropy(x, y):
    """
    calculate SA ST SB AT AB TB
    calculate by bars
    plot the cross entropy between the two parts
    input: two parts of choir
    output: plot
    """
    softmax = nn.Softmax(dim=-1)
    y = torch.argmax(softmax(y), dim=-1)
    mask = (x != TOKEN_PAD)
    y = y[mask]
    x = x[mask]
    y = y.cpu().numpy()
    x = x.cpu().numpy()

    S_y, A_y, T_y, B_y = yseq2SATB(y)
    S_x, A_x, T_x, B_x = xseq2SATB(x)

    p_sx, p_ax, p_tx, p_bx = prob_distri(S_x), prob_distri(A_x), prob_distri(T_x), prob_distri(B_x)
    p_sy, p_ay, p_ty, p_by = prob_distri(S_y), prob_distri(A_y), prob_distri(T_y), prob_distri(B_y)

    H_SA = vocal_part_cross_entropy(p_sx, p_ax, p_ay)/len(p_sx)
    H_ST = vocal_part_cross_entropy(p_sx, p_tx, p_ty)/len(p_sx)
    H_SB = vocal_part_cross_entropy(p_sx, p_bx, p_by)/len(p_sx)
    H_AT = vocal_part_cross_entropy(p_ax, p_tx, p_ty)/len(p_sx)
    H_AB = vocal_part_cross_entropy(p_ax, p_bx, p_by)/len(p_sx)
    H_TB = vocal_part_cross_entropy(p_tx, p_bx, p_by)/len(p_sx)
    return (H_SA + H_ST + H_SB + H_AT + H_AB + H_TB)


def vocal_part_cross_entropy(x1, y1, y2):
    '''
    input:x1, y1, y2
    f(x1,y1,y2) = cross_entropy(x1, y1) - cross_entropy(x1, y2)
    '''
    return cross_entropy(x1, y1) - cross_entropy(x1, y2)

def prob_distri(seq):
    '''
    input: a seq of notes.
    calculate probability distribution
    output: a seq of probability distribution
    '''
    output = np.ones(129)
    for i in seq:
        output[i] += 1
    return output / np.sum(seq)


def cross_entropy(x, y):
    '''
    calculate x , y cross entropy
    input ; x, y
    - sum (x * log2(y))
    output value of bar
    '''
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    if torch.is_tensor(y):
        y = y.cpu().numpy()
    # assert x == y
    return -np.sum(x * np.log2(y))

def entropy(x):
    '''
    calculate x  entropy
    input ; x
    - sum (x * log2(x))
    '''
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    return -np.sum(x * np.log2(x))


def yseq2SATB(y):
    '''
    input : seq of notes from dataset
    output : SATB of dataset and SATB of model create
    '''
    S, A, T, B = [], [], [], []
    for k, v in enumerate(y):
        if k % 5 == 0:
            S.append(v)
        if k % 5 == 1:
            A.append(v)
        if k % 5 == 2:
            T.append(v)
        if k % 5 == 3:
            B.append(v)
    return S, A, T, B


def xseq2SATB(x):
    '''
    input : seq of notes from dataset
    output : SATB of dataset and SATB of model create
    '''
    S, A, T, B = [], [], [] ,[]
    for k, v in enumerate(x):
        if k % 5 == 1:
            S.append(v)
        if k % 5 == 2:
            A.append(v)
        if k % 5 == 3:
            T.append(v)
        if k % 5 == 4:
            B.append(v)
    return S,A,T,B

def calculate_rhy_consistency(x, y):
    '''
    calculate rhythm between two vocal parts:
    input: x part & y part
    output:
    '''
    softmax = nn.Softmax(dim=-1)
    y = torch.argmax(softmax(y), dim=-1)
    mask = (x != TOKEN_PAD)
    y = y[mask]
    x = x[mask]

    S_x, A_x, T_x, B_x = xseq2SATB(x.cpu().numpy())
    S_x_diff = np.array([1 if i != 0 else 0 for i in np.diff(np.array(S_x))])
    A_x_diff = np.array([1 if i != 0 else 0 for i in np.diff(np.array(A_x))])
    T_x_diff = np.array([1 if i != 0 else 0 for i in np.diff(np.array(T_x))])
    B_x_diff = np.array([1 if i != 0 else 0 for i in np.diff(np.array(B_x))])
    S_y, A_y, T_y, B_y = yseq2SATB(y.cpu().numpy())
    # calculate diff
    S_y_diff = np.array([1 if i != 0 else 0 for i in np.diff(np.array(S_y))])
    A_y_diff = np.array([1 if i != 0 else 0 for i in np.diff(np.array(A_y))])
    T_y_diff = np.array([1 if i != 0 else 0 for i in np.diff(np.array(T_y))])
    B_y_diff = np.array([1 if i != 0 else 0 for i in np.diff(np.array(B_y))])

    seq_len = S_x_diff.shape[0]
    rhy_consistency = np.sum(S_x_diff * A_y_diff + S_x_diff * T_y_diff + S_x_diff * B_y_diff + A_y_diff * T_y_diff + \
                      A_y_diff * B_y_diff + T_y_diff * B_y_diff)
    return rhy_consistency / seq_len





