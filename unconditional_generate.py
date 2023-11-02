'''
test for the length of generation
with rpr or without rpr
'''
import numpy as np
import torch
import torch.nn as nn
import os
import random
import math
import matplotlib.pyplot as plt
import matplotlib

from utilities.argument_funcs import parse_generate_args, print_generate_args
from model.CoCoFormer import CoCoformer
from dataset.jsf import create_jsf_datasets, compute_jsf_accuracy, process_midi
from torch.utils.data import DataLoader
from torch.optim import Adam
import mido


from utilities.constants import *
from utilities.device import get_device, use_cuda
from utilities.argument_funcs import parse_generate_args
from utilities.processor import encode_midi

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
    for key, value in enumerate(index):
        if key % 5 == 1:
            s.append(value)
        if key % 5 == 2:
            a.append(value)
        if key % 5 == 3:
            t.append(value)
        if key % 5 == 4:
            b.append(value)
    s = [i if i < 128 else 127 for i in s]
    a = [i if i < 128 else 127 for i in a]
    t = [i if i < 128 else 127 for i in t]
    b = [i if i < 128 else 127 for i in b]

    if not single:

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
        # TODO: need to mix four track
        # create_track(track, s)
        # create_track(track, a)
        # create_track(track, t)
        # create_track(track, b)
    mid.save(file_path)


# main
def unconditional_generate():

    print("start unconditional generate ...")
    args = parse_generate_args()
    print_generate_args(args)

    if args.force_cpu:
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    os.makedirs(args.output_dir, exist_ok=True)

    # Grabbing dataset if needed
    # _, _, dataset = create_epiano_datasets(args.midi_root, args.num_prime, random_seq=False)
    _, dataset = create_jsf_datasets(args.midi_root, args.num_prime, random_seq=False)
    # Can be None, an integer index to dataset, or a file path
    if(args.primer_file is None):
        f = str(random.randrange(len(dataset)))
    else:
        f = args.primer_file

    if(f.isdigit()):
        idx = int(f)
        primer, _  = dataset[idx]
        primer = primer.to(get_device())

        print("Using primer index:", idx, "(", dataset.data_files[idx], ")")

    else:
        raw_mid = encode_midi(f)
        if(len(raw_mid) == 0):
            print("Error: No midi messages in primer file:", f)
            return

        primer, _  = process_midi(raw_mid, args.num_prime, random_seq=False)
        primer = torch.tensor(primer, dtype=TORCH_LABEL_TYPE, device=get_device())

        print("Using primer file:", f)

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
                             event2word=event2word).to(get_device())


    model.load_state_dict(torch.load(args.model_weights))

    # Saving primer first
    f_path = os.path.join(args.output_dir, "primer.mid")
    decode(primer[:args.num_prime].cpu().numpy(), file_path=f_path)

    # GENERATION
    model.eval()
    with torch.set_grad_enabled(False):
        if(args.beam > 0):
            print("BEAM:", args.beam)
            beam_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=args.beam)

            f_path = os.path.join(args.output_dir, "beam.mid")
            decode(beam_seq[0].cpu().numpy(), file_path=f_path)
        else:
            print("RAND DIST")
            rand_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=0)

            f_path = os.path.join(args.output_dir, "rand.mid")
            decode(rand_seq[0].cpu().numpy(), file_path=f_path)
            # plot pianoroll
            plot_pianoroll(rand_seq[0].cpu().numpy())


def plot_pianoroll(seq):
    '''
    plot painoroll
    input : seqs of words
    output : a image of painoroll
    '''
    # build matrix
    s, a, t, b = [], [], [], []
    for key, value in enumerate(seq):
        if key % 5 == 1:
            s.append(value)
        if key % 5 == 2:
            a.append(value)
        if key % 5 == 3:
            t.append(value)
        if key % 5 == 4:
            b.append(value)
    pianoroll = np.ones((180, 500))

    def plot_track(seq):
        for k, v in enumerate(seq):
            pianoroll[v, k] = 0
    plot_track(s)
    plot_track(a)
    plot_track(t)
    plot_track(b)
    pianoroll = np.flip(pianoroll[30:100], axis=0)
    cmp = matplotlib.colors.ListedColormap(['g', 'w'])
    plt.figure(1)
    plt.imshow(pianoroll, cmap=cmp)
    plt.show()


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
    unconditional_generate()
