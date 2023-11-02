import torch
import torch.nn as nn
import os
import random
import math

from utilities.argument_funcs import parse_generate_args, print_generate_args
from model.CoCoFormer import CoCoformer
from dataset.jsf import create_jsf_datasets, compute_jsf_accuracy, process_midi
from torch.utils.data import DataLoader
from torch.optim import Adam
import mido
import music21
from utilities.constants import *
from utilities.device import get_device, use_cuda
from utilities.argument_funcs import parse_generate_args

##### read word2event event2word
args = parse_generate_args()
f = open(args.word2event, 'rb')
word2event = pickle.load(f)
# reverse the vector event2word
event2word = {}
for key, val in word2event.items():
    event2word[val] = key


def create_track(track, seq):
    '''
    create a midi track of seq
    '''
    note = 0
    time = 120
    for i in seq:
        if note != int(i):
            if note == 0:
                note = i
                track.append(mido.Message('note_on', note=note, velocity=96, time=0))
            else:
                track.append(mido.Message('note_off', note=note, velocity=96, time=time))
                note = i
                time = 120
                track.append(mido.Message('note_on', note=note, velocity=96, time=0))
        else:
            time += 120
    track.append(mido.Message('note_off', note=note, velocity=96, time=time))


def decode(index, file_path, single=False):

    event = [word2event[i] for i in index]
    print("decoding...")
    s, a, t, b = [], [], [], []
    if not single:
        for key, value in enumerate(index):
            if key % 9 == 2:
                assert value < 129
                s.append(value)
                continue
            if key % 9 == 4:
                assert value < 129
                a.append(value)
                continue
            if key % 9 == 6:
                assert value < 129
                t.append(value)
                continue
            if key % 9 == 8:
                assert value < 129
                b.append(value)
                continue

        mid = mido.MidiFile()
        track_s = mido.MidiTrack()
        mid.tracks.append(track_s)
        create_track(track_s, s)

        track_a = mido.MidiTrack()
        mid.tracks.append(track_a)
        create_track(track_a, a)

        track_t = mido.MidiTrack()
        mid.tracks.append(track_t)
        create_track(track_t, t)

        track_b = mido.MidiTrack()
        mid.tracks.append(track_b)
        create_track(track_b, b)
    else:
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        create_track(track, seq)
    mid.save(file_path)
    plot_pianoroll(s, a, t, b)
    print("midi save in:", file_path)

def plot_pianoroll(s, a, t, b):
    '''
    plot painoroll
    input : seqs of words
    output : a image of painoroll
    '''
    # build matrix
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    pianoroll = np.ones((180, 500))

    def plot_track(seq):
        for k, v in enumerate(seq):
            pianoroll[v, k] = 0

    def plot_main_track(seq):
        for k, v in enumerate(seq):
            pianoroll[v, k] = 2

    plot_main_track(s)
    plot_track(a)
    plot_track(t)
    plot_track(b)
    pianoroll = np.flip(pianoroll[30:100], axis=0)
    cmp = matplotlib.colors.ListedColormap(['g', 'w', 'b'])
    plt.figure(1)
    plt.imshow(pianoroll, cmap=cmp)
    plt.show()

def conditional_generate(seq, chord, bs, ba, bt, bb):

    assert len(seq) == len(chord) == len(bs) == len(ba) == len(bt) == len(bb)
    beats = []
    for i in range(len(bs)):
        beats.extend((bs[i], ba[i], bt[i], bb[i]))
    # deal with input: slice it < 128
    input_note, input_chord, input_beats, input_bs, input_ba, input_bt, input_bb = [], [], [], [], [], [], []
    loop = int(math.ceil(len(seq)/64))
    for i in range(loop):
        if i+64 <= len(seq):
            input_note.append(seq[i*64: (i+1)*64])
            input_chord.append(chord[i*64: (i+1)*64])
            input_bs.append(bs[i*64: (i+1)*64])
            input_ba.append(ba[i*64: (i+1)*64])
            input_bt.append(bt[i*64: (i+1)*64])
            input_bb.append(bb[i*64: (i+1)*64])
        else:
            input_note.append(seq[i:len(seq)])
            input_chord.append(chord[i:len(seq)])
            input_bs.append(bs[i:len(seq)])
            input_ba.append(ba[i:len(seq)])
            input_bt.append(bt[i:len(seq)])
            input_bb.append(bb[i:len(seq)])

    for p in range(len(input_bs)):
        b = []
        for q in range(len(input_bs[0])):
            b.extend((input_bs[p][q], input_ba[p][q], input_bt[p][q], input_bb[p][q]))
        input_beats.append(b)
    args = parse_generate_args()
    print_generate_args(args)

    print("start conditional generate ...")
    if args.force_cpu:
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    os.makedirs(args.conditional_output_dir, exist_ok=True)

    primer = torch.tensor([chord[0], seq[0]])
    ##### read word2event event2word
    f = open(args.word2event, 'rb')
    word2event = pickle.load(f)
    # reverse the vector event2word
    event2word = {}
    for key, val in word2event.items():
        event2word[val] = key

    model = CoCoformer(n_layers=args.n_layers, num_heads=args.num_heads,
                             d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                             max_sequence=args.max_sequence, rpr=args.rpr, word2event=word2event,
                             event2word=event2word)

    if args.gpu[0] != -1:
        model = nn.DataParallel(model, device_ids=args.gpu)
        model.cuda(device=args.gpu[0])
    # model.load_state_dict(torch.load(args.model_weights))
    model.load_state_dict(
        torch.load(args.model_weights, map_location=lambda storage, loc: storage.cuda(device=args.gpu[0])),
        strict=False)

    # Saving primer first
    f_path = os.path.join(args.conditional_output_dir, "primer.mid")
    decode(seq, file_path=f_path, single=True)

    # GENERATION
    model.eval()
    with torch.set_grad_enabled(False):
        if (args.beam > 0):
            print("BEAM:", args.beam)
            beam_seq = model.conditional_generate(seq, args.target_seq_length, beam=args.beam)

            f_path = os.path.join(args.output_dir, "beam.mid")
            decode(beam_seq[0].cpu().numpy(), file_path=f_path)
        else:
            print("RAND DIST")
            rand_list = []
            n = 0
            while n < loop:
                flag = model.module.conditional_generate(input_beats[n], input_chord[n],
                                                         input_note[n], input_chord[n], input_bs[n], input_ba[n],
                                                         input_bt[n], input_bb[n], args.target_seq_length, beam=0)
                if flag is False:
                    continue
                else:
                    rand_list.append(flag)
                    n += 1
            rand_seq = torch.cat(rand_list, 1)
            f_path = os.path.join(args.output_dir, "output.mid")
            decode(rand_seq[0].cpu().numpy(), file_path=f_path)


def midi2seq(input_midi):
    s = []
    input_midi = input_midi.tracks[1]
    for i in input_midi:
        if 'note_off' in i.type:
            s.append([i.note] * int(i.time/240))
    seq = [j for k in s for j in k]
    return seq

def chord2word(s):
    ##### read word2event event2word
    f = open(args.word2event, 'rb')
    word2event = pickle.load(f)
    # reverse the vector event2word
    event2word = {}
    for key, val in word2event.items():
        event2word[val] = key
    return [event2word[i] for i in s]

if __name__ == "__main__":
    # unconditional_generate()
    input_midi = mido.MidiFile('./generate/input1.mid')
    seq = midi2seq(input_midi)
    print("seqs len:", len(seq))

    # for input_1.mid
    chord_seq = ['C']*4*8 + ['E']*4*8 + ['C']*4*8 + ['E']*4*4 + ['D']*4*4

    beat_s = ['1'] + ['_']*3 + ['1'] + ['_'] + ['1'] + ['_'] + ['1'] + ['_']*3 + ['1'] + ['_'] + ['1'] + ['_'] \
             +['1'] + ['_'] + ['1'] + ['_'] + ['1'] + ['_']*3 + ['1'] + ['_'] + ['1'] + ['_'] + ['1'] + ['_'] + ['1'] + ['_'] \
             + ['1'] + ['_']*3 + ['1'] + ['_'] + ['1'] + ['_'] + ['1'] + ['_']*3 + ['1'] + ['_'] + ['1'] + ['_'] \
             + ['1'] + ['_'] + ['1'] + ['_'] + ['1'] + ['_']*3 + ['1'] + ['_'] + ['1'] + ['_'] + ['1'] + ['_'] + ['1'] + ['_'] \
             + ['1'] + ['_']*3 + ['1'] + ['_'] + ['1'] + ['_'] + ['1'] + ['_']*3 + ['1'] + ['_'] + ['1'] + ['_']\
             + ['1'] + ['_'] + ['1'] + ['_'] + ['1'] + ['_']*3 + ['1'] + ['_'] + ['1'] + ['_'] + ['1'] + ['_'] + ['1'] + ['_']\
             + ['1'] + ['_'] + ['1'] + ['_'] + ['1'] + ['_'] + ['1'] + ['_'] + ['1'] + ['_']*3 + ['1'] + ['_']*3 \
             + ['1'] + ['_'] + ['1'] + ['_'] + ['1'] + ['_']*3 + ['1'] + ['_'] + ['1'] + ['_'] + ['1'] + ['_'] + ['1'] + ['_']
    # beat_s = ['1'] + ['_']*3 + ['_']*4*7 + ['1'] + ['_']*3 + ['_']*4*7 + ['1'] + ['_']*3 + ['_']*4*7 + ['1'] + ['_']*3 + ['_']*4*7
    # beat_a = ['1'] + ['_']*3 + ['_']*4*7 + ['1'] + ['_']*3 + ['_']*4*7 + ['1'] + ['_']*3 + ['_']*4*7 + ['1'] + ['_']*3 + ['_']*4*7
    # beat_t = ['1'] + ['_']*3 + ['_']*4*7 + ['1'] + ['_']*3 + ['_']*4*7 + ['1'] + ['_']*3 + ['_']*4*7 + ['1'] + ['_']*3 + ['_']*4*7
    # beat_b = ['1'] + ['_']*3 + ['_']*4*7 + ['1'] + ['_']*3 + ['_']*4*7 + ['1'] + ['_']*3 + ['_']*4*7 + ['1'] + ['_']*3 + ['_']*4*7
    beat_a = beat_s
    beat_t = beat_s
    beat_b = beat_s
    beat_dict = {'1': 'start', '0': 'stop', '_': 'hold'}
    bs = [event2word['S_' + beat_dict[i]] for i in beat_s]
    ba = [event2word['A_' + beat_dict[i]] for i in beat_a]
    bt = [event2word['T_' + beat_dict[i]] for i in beat_t]
    bb = [event2word['B_' + beat_dict[i]] for i in beat_b]


    chord = chord2word(chord_seq)
    print("chord len:", len(chord))
    conditional_generate(seq, chord, bs, ba, bt, bb)
