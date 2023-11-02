# -*-coding:utf-8-*-
# read the dataset and turn it into pickle
# C S A T B
import argparse
import copy
import os
import numpy as np
import shutil
import pickle
import tqdm
import mido


def build_event2word():
    '''
    build midi-events from datasets
    1) Track_start Track_end
    2) Bar
    3) Note-On
    4) Note-Velocity
    5) Note-Duration
    6) Position
    7) Chord
    '''

    event2word = {}
    # Note-on
    # 0~127 means note on
    # 128 means note stop
    for i in range(128):
        key = 'Note-On_' + str(i)
        event2word[key] = i
    event2word['Note-Stop'] = 128
    # Chord build
    chord = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B',
             'Cm', 'Cm#', 'Dm', 'Emb', 'Em', 'Fm', 'Fm#', 'Gm', 'Amb', 'Am', 'Bmb', 'Bm',
             'Cdim', 'Cdim#', 'Ddim', 'Edimb', 'Edim', 'Fdim', 'Fdim#', 'Gdim', 'Adimb', 'Adim', 'Bdimb', 'Bdim',
             'Caug', 'Caug#', 'Daug', 'Eaugb', 'Eaug', 'Faug', 'Faug#', 'Gaug', 'Aaugb', 'Aaug', 'Baugb', 'Baug',
             'other', 'rest']
    for i in chord:
        event2word[i] = len(event2word)
    # add end other
    # event2word['other'] = len(event2word)
    # event2word['rest'] = len(event2word)
    # beat and hold:
    event2word['S_start'] = 179
    event2word['S_stop'] = 180
    event2word['S_hold'] = 181

    event2word['A_start'] = 182
    event2word['A_stop'] = 183
    event2word['A_hold'] = 184

    event2word['T_start'] = 185
    event2word['T_stop'] = 186
    event2word['T_hold'] = 187

    event2word['B_start'] = 188
    event2word['B_stop'] = 189
    event2word['B_hold'] = 190

    f = open('./event2word.pkl', 'wb')
    pickle.dump(event2word, f)
    f.close()

    return event2word


def build_word2event(event2word):
    # reverse the vector event--rpr -output_dir "./baseline_test_rpr" --gpu 0 12word
    word2event = {}
    for key, val in event2word.items():
        word2event[val] = key

    f = open('./word2event.pkl', 'wb')
    pickle.dump(word2event, f)
    f.close()

    return word2event


# def process(root, args, word2event, valid_p=0.1, test_p=0.2):
#
#     print("processing JSF data...")
#     data_ori = np.load(root, allow_pickle=True, encoding='latin1')
#     data_pitches = data_ori['pitches']
#     data_chords = data_ori['chords']
#     # build dataset:
#     seqs_bar = []
#     seqs_bar_event = []
#     seqs_bar_amp = []
#     seqs_bar_event_amp = []
#     seqs_bar_rev = []
#     seqs_bar_event_rev = []
#     if args.arrangement == 0:
#         print("processing origin data...")
#         for k, v in enumerate(tqdm.tqdm(data_pitches)):
#             seq_bar = []
#             seq_bar_event = []
#             for k_s, v_s in enumerate(v):
#                 seq_bar.append(129 + int(data_chords[k][k_s]))
#                 seq_bar_event.append(word2event[129 + int(data_chords[k][k_s])])
#                 for j in range(4):
#                     if int(v_s[j]) != -1:
#                         seq_bar.append(int(v_s[j]))
#                         seq_bar_event.append(word2event[int(v_s[j])])
#                     else:
#                         seq_bar.append(128)
#                         seq_bar_event.append(word2event[128])
#             seqs_bar.append(seq_bar)
#             seqs_bar_event.append(seq_bar_event)
#         if not args.data_amplification and not args.data_reversal:
#             return seqs_bar, seqs_bar_event
#         # data with amplification
#         if args.data_amplification:
#             print("amplification data...")
#             for k, v in enumerate(tqdm.tqdm(data_pitches)):
#                 for i in range(-5, 7):  # aug
#                     seq_bar_amp = []
#                     seq_bar_event_amp = []
#                     for k_s, v_s in enumerate(v):
#                         if int(data_chords[k][k_s]) + i < 0:
#                             seq_bar_amp.append(129 + int(data_chords[k][k_s]) + (12 + i))
#                             seq_bar_event_amp.append(word2event[129 + int(data_chords[k][k_s]) + (12 + i)])
#                         elif int(data_chords[k][k_s]) + i + 129 > 176:
#                             seq_bar_amp.append(129 + int(data_chords[k][k_s]) - (12 - i))
#                             seq_bar_event_amp.append(word2event[129 + int(data_chords[k][k_s]) - (12 - i)])
#                         else:
#                             seq_bar_amp.append(129 + int(data_chords[k][k_s]) + i)
#                             seq_bar_event_amp.append(word2event[129 + int(data_chords[k][k_s]) + i])
#                         for j in range(4):
#                             if int(v_s[j]) != -1:
#                                 seq_bar_amp.append(int(v_s[j]) + i)
#                                 seq_bar_event_amp.append(word2event[int(v_s[j]) + i])
#                             else:
#                                 seq_bar_amp.append(128)
#                                 seq_bar_event_amp.append(word2event[128])
#                     seqs_bar_amp.append(seq_bar_amp)
#                     seqs_bar_event_amp.append(seq_bar_event_amp)
#             if args.data_amplification and args.data_reversal == False:
#                 return seqs_bar_amp, seqs_bar_event_amp
#         if args.data_reversal:
#             print("start reverse the dataset...")
#             for k, v in enumerate(tqdm.tqdm(data_pitches)):
#                 for i in range(-5, 7):  # aug
#                     seq_bar_rev = []
#                     seq_bar_event_rev = []
#                     for k_s, v_s in enumerate(v[::-1]):
#                         if int(data_chords[k][k_s]) + i < 0:
#                             seq_bar_rev.append(129 + int(data_chords[k][k_s]) + (12 + i))
#                             seq_bar_event_rev.append(word2event[129 + int(data_chords[k][k_s]) + (12 + i)])
#                         elif int(data_chords[k][k_s]) + i + 129 > 176:
#                             seq_bar_rev.append(129 + int(data_chords[k][k_s]) - (12 - i))
#                             seq_bar_event_rev.append(word2event[129 + int(data_chords[k][k_s]) - (12 - i)])
#                         else:
#                             seq_bar_rev.append(129 + int(data_chords[k][k_s]) + i)
#                             seq_bar_event_rev.append(word2event[129 + int(data_chords[k][k_s]) + i])
#                         for j in range(4):
#                             if int(v_s[j]) != -1:
#                                 seq_bar_rev.append(int(v_s[j]) + i)
#                                 seq_bar_event_rev.append(word2event[int(v_s[j]) + i])
#                             else:
#                                 seq_bar_rev.append(128)
#                                 seq_bar_event_rev.append(word2event[128])
#                     seqs_bar_rev.append(seq_bar_rev)
#                     seqs_bar_event_rev.append(seq_bar_event_rev)
#             for _ in seqs_bar_amp:
#                 seqs_bar_rev.append(_)
#             for _ in seqs_bar_event_amp:
#                 seqs_bar_event_rev.append(_)
#             return seqs_bar_rev, seqs_bar_event_rev


def process_SATB(root, word2event, event2word):
    '''
    process dataset for models:
    1) SATB inputs for backbone1
    2) chord inputs for backbone2
    '''
    # input the data from pkl
    print("processing JSF data...")
    data_ori = np.load(root, allow_pickle=True, encoding='latin1')
    data_pitches = data_ori['pitches']
    # make the same note as hold-on marks:
    # build STAB:
    seqs_bar = []
    seqs_bar_event = []
    for d in data_pitches:
        seqs_bar.append(d.reshape(-1).astype('int16'))

    print("convert to seqs format...")
    for k, v in enumerate(tqdm.tqdm(seqs_bar)):
        seq_bar_event = []
        for k_s, v_s in (enumerate(v)):
            if v_s != -1:
                seq_bar_event.append(word2event[v_s])
            else:
                seqs_bar[k][k_s] = 128
                seq_bar_event.append(word2event[128])
        seqs_bar_event.append(seq_bar_event)

    # build chord
    data_chords = data_ori['chords']
    # change to list
    seqs_chord = []
    seqs_chord_event = []
    print("build chords...")
    for i in tqdm.tqdm(list(data_chords)):
        seq_chord = []
        seq_chord_event = []
        for j in i:
            seq_chord.append(j+129)
            seq_chord_event.append(word2event[j + 129])
        seqs_chord_event.append(seq_chord_event)
        seqs_chord.append(seq_chord)

    # build rhythm and hold mark
    beat_all, beat_s_event, beat_a_event, beat_t_event, beat_b_event = build_beat(seqs_bar, event2word, word2event)
    # merge_seqs = [(n, c, s, a, t, b) for n, c, s, a, t, b in zip(seqs_bar, seqs_chord, beat_s, beat_a, beat_t, beat_b)]
    merge_seqs = [(n, c, b) for n, c, b in zip(seqs_bar, seqs_chord, beat_all)]
    merge_seqs_event = [(n, c, s, a, t, b) for n, c, s, a, t, b in zip(seqs_bar_event, seqs_chord_event, beat_s_event,
                                                                       beat_a_event, beat_t_event, beat_b_event)]
    return merge_seqs, merge_seqs_event


def build_beat(seqs_bar, event2word, word2event):
    beat_s, beat_a, beat_t, beat_b = [], [], [], []
    beat_s_event, beat_a_event, beat_t_event, beat_b_event = [], [], [], []
    beat_all = []
    beat_all_event = []
    for k, v in enumerate(tqdm.tqdm(seqs_bar)):
        b_s, b_a, b_t, b_b = [], [], [], []
        b_s_e, b_a_e, b_t_e, b_b_e = [], [], [], []
        b_all = []
        b_all_event = []
        # define the first value: start or stop mark:
        past_s = v[0]  # define past_s as the first element
        if past_s != event2word['Note-Stop']:  # if not stop-note, Add start note
            b_s.append(event2word['S_start'])
            b_all.append(event2word['S_start'])
            b_all_event.append(word2event[event2word['S_start']])
            b_s_e.append(word2event[event2word['S_start']])
        else:
            b_s.append(event2word['S_stop'])
            b_all.append(event2word['S_stop'])
            b_all_event.append(word2event[event2word['S_stop']])
            b_s_e.append(word2event[event2word['S_stop']])

        past_a = v[1]
        if past_a != event2word['Note-Stop']:
            b_a.append(event2word['A_start'])
            b_all.append(event2word['A_start'])
            b_all_event.append(word2event[event2word['A_start']])
            b_a_e.append(word2event[event2word['A_start']])
        else:
            b_a.append(event2word['A_stop'])
            b_all.append(event2word['A_stop'])
            b_all_event.append(word2event[event2word['A_stop']])
            b_a_e.append(word2event[event2word['A_stop']])

        past_t = v[2]
        if past_t != event2word['Note-Stop']:
            b_t.append(event2word['T_start'])
            b_all.append(event2word['T_start'])
            b_all_event.append(word2event[event2word['T_start']])
            b_t_e.append(word2event[event2word['T_start']])
        else:
            b_t.append(event2word['T_stop'])
            b_all.append(event2word['T_stop'])
            b_all_event.append(word2event[event2word['T_stop']])
            b_t_e.append(word2event[event2word['T_stop']])

        past_b = v[3]
        if past_b != event2word['Note-Stop']:
            b_b.append(event2word['B_start'])
            b_all.append(event2word['B_start'])
            b_all_event.append(word2event[event2word['B_start']])
            b_b_e.append(word2event[event2word['B_start']])
        else:
            b_b.append(event2word['B_stop'])
            b_all.append(event2word['B_stop'])
            b_all_event.append(word2event[event2word['B_stop']])
            b_b_e.append(word2event[event2word['B_stop']])

        index = 4
        while index < v.shape[0]:
            if index % 4 == 0:
                # if v is not stop-note, judge if v is or not the past, if yes, start, if not, hold
                if v[index] != event2word['Note-Stop']:
                    if v[index] != past_s:
                        past_s = v[index]
                        b_s.append(event2word['S_start'])
                        b_all.append(event2word['S_start'])
                        b_all_event.append(word2event[event2word['S_start']])
                        b_s_e.append(word2event[event2word['S_start']])
                    else:
                        b_s.append(event2word['S_hold'])
                        b_all.append(event2word['S_hold'])
                        b_all_event.append(word2event[event2word['S_hold']])
                        b_s_e.append(word2event[event2word['S_hold']])
                else:
                    b_s.append(event2word['S_stop'])
                    b_all.append(event2word['S_stop'])
                    b_all_event.append(word2event[event2word['S_stop']])
                    b_s_e.append(word2event[event2word['S_stop']])
                index += 1
                continue
            if index % 4 == 1:
                if v[index] != event2word['Note-Stop']:
                    if v[index] != past_a:
                        past_a = v[index]
                        b_a.append(event2word['A_start'])
                        b_all.append(event2word['A_start'])
                        b_all_event.append(word2event[event2word['A_start']])
                        b_a_e.append(word2event[event2word['A_start']])
                    else:
                        b_a.append(event2word['A_hold'])
                        b_all.append(event2word['A_hold'])
                        b_all_event.append(word2event[event2word['A_hold']])
                        b_a_e.append(word2event[event2word['A_hold']])
                else:
                    b_a.append(event2word['A_stop'])
                    b_all.append(event2word['A_stop'])
                    b_all_event.append(word2event[event2word['A_stop']])
                    b_a_e.append(word2event[event2word['A_stop']])
                index += 1
                continue
            if index % 4 == 2:
                if v[index] != event2word['Note-Stop']:
                    if v[index] != past_t:
                        past_t = v[index]
                        b_t.append(event2word['T_start'])
                        b_all.append(event2word['T_start'])
                        b_all_event.append(word2event[event2word['T_start']])
                        b_t_e.append(word2event[event2word['T_start']])
                    else:
                        b_t.append(event2word['T_hold'])
                        b_all.append(event2word['T_hold'])
                        b_all_event.append(word2event[event2word['T_hold']])
                        b_t_e.append(word2event[event2word['T_hold']])
                else:
                    b_t.append(event2word['T_stop'])
                    b_all.append(event2word['T_stop'])
                    b_all_event.append(word2event[event2word['T_stop']])
                    b_t_e.append(word2event[event2word['T_stop']])
                index += 1
                continue
            if index % 4 == 3:
                if v[index] != event2word['Note-Stop']:
                    if v[index] != past_b:
                        past_b = v[index]
                        b_b.append(event2word['B_start'])
                        b_all.append(event2word['B_start'])
                        b_all_event.append(word2event[event2word['B_start']])
                        b_b_e.append(word2event[event2word['B_start']])
                    else:
                        b_b.append(event2word['B_hold'])
                        b_all.append(event2word['B_hold'])
                        b_all_event.append(word2event[event2word['B_hold']])
                        b_b_e.append(word2event[event2word['B_hold']])
                else:
                    b_b.append(event2word['B_stop'])
                    b_all.append(event2word['B_stop'])
                    b_all_event.append(word2event[event2word['B_stop']])
                    b_b_e.append(word2event[event2word['B_stop']])
                index += 1
                continue
        beat_s.append(b_s)
        beat_s_event.append(b_s_e)

        beat_a.append(b_a)
        beat_a_event.append(b_a_e)

        beat_t.append(b_t)
        beat_t_event.append(b_t_e)

        beat_b.append(b_b)
        beat_b_event.append(b_b_e)

        beat_all.append(b_all)
        beat_all_event.append(b_all_event)

    return beat_all, beat_s_event, beat_a_event, beat_t_event, beat_b_event


def process_midi_CBSATB(midi_dir, chord_dir, word2event, event2word):

    midi_list = os.listdir(midi_dir)
    midi_list.sort(key=lambda x:int(x[:-4]))
    pitches = []
    for k, i in enumerate(tqdm.tqdm(midi_list)):
        mid = mido.MidiFile(midi_dir + '/' + i)
        s = process_midi_track(mid, 1)  # 1 means track
        a = process_midi_track(mid, 2)
        t = process_midi_track(mid, 3)
        b = process_midi_track(mid, 4)
        # merge SATB in same t as [S0, A0, T0, B0, ..., Sn, An, Tn, Bn]
        assert len(s) == len(a) == len(t) == len(b)
        pitch = []
        for j in range(len(s)):
            pitch.append(s[j])
            pitch.append(a[j])
            pitch.append(t[j])
            pitch.append(b[j])
        pitches.append(np.array(pitch))

    # build chord
    data_ori = np.load(chord_dir, allow_pickle=True, encoding='latin1')
    data_chords = data_ori['chords']
    data_pitches = data_ori['pitches']
    pitches_pkl = []
    for d in data_pitches:
        pitches_pkl.append(d.reshape(-1).astype('int16'))
    # change to list
    chord = []
    chord_event = []
    print("build chords...")
    for i in tqdm.tqdm(list(data_chords)):
        seq_chord = []
        seq_chord_event = []
        for j in i:
            seq_chord.append(j+129)
            seq_chord_event.append(word2event[j + 129])
        chord_event.append(seq_chord_event)
        chord.append(seq_chord)

    # build beat
    beats = []
    for k, i in enumerate(tqdm.tqdm(midi_list)):
        mid = mido.MidiFile(midi_dir + '/' + i)
        s = midi2beat(mid, 'S', event2word)  # 1 means track
        a = midi2beat(mid, 'A', event2word)
        t = midi2beat(mid, 'T', event2word)
        b = midi2beat(mid, 'B', event2word)
        # merge SATB in same t as [S0, A0, T0, B0, ..., Sn, An, Tn, Bn]
        assert len(s) == len(a) == len(t) == len(b)
        beat = []
        for j in range(len(s)):
            beat.append(s[j])
            beat.append(a[j])
            beat.append(t[j])
            beat.append(b[j])
        beats.append(np.array(beat))

    # add chord beat in pitches: C0 Bs0 S0 Ba0 A0 Bt0 T0 Bb0 B0
    print("add chord beat in pitches:")
    pitches_with_chord_beat = []
    for k in range(len(chord)):
        pitch_with_chord_beat = []
        for k_s in range(len(chord[k])):
            pitch_with_chord_beat.append(chord[k][k_s])
            for i in range(4):
                pitch_with_chord_beat.append(beats[k][k_s*4 + i])
                pitch_with_chord_beat.append(pitches[k][k_s*4 + i])
        pitches_with_chord_beat.append(pitch_with_chord_beat)

    merge_seqs = [(n, c, b) for n, c, b in zip(pitches_with_chord_beat, chord, beats)]
    return merge_seqs


def process_midi_SATB(midi_dir, chord_dir, word2event, event2word):

    midi_list = os.listdir(midi_dir)
    pitches = []
    for k, i in enumerate(tqdm.tqdm(midi_list)):
        mid = mido.MidiFile(midi_dir + '/' + i)
        s = process_midi_track(mid, 1)  # 1 means track
        a = process_midi_track(mid, 2)
        t = process_midi_track(mid, 3)
        b = process_midi_track(mid, 4)
        # merge SATB in same t as [S0, A0, T0, B0, ..., Sn, An, Tn, Bn]
        assert len(s) == len(a) == len(t) == len(b)
        pitch = []
        for j in range(len(s)):
            pitch.append(s[j])
            pitch.append(a[j])
            pitch.append(t[j])
            pitch.append(b[j])
        pitches.append(np.array(pitch))

    # build chord
    data_ori = np.load(chord_dir, allow_pickle=True, encoding='latin1')
    data_chords = data_ori['chords']
    # change to list
    chord = []
    chord_event = []
    print("build chords...")
    for i in tqdm.tqdm(list(data_chords)):
        seq_chord = []
        seq_chord_event = []
        for j in i:
            seq_chord.append(j+129)
            seq_chord_event.append(word2event[j + 129])
        chord_event.append(seq_chord_event)
        chord.append(seq_chord)

    # build beat
    beats = []
    for k, i in enumerate(tqdm.tqdm(midi_list)):
        mid = mido.MidiFile(midi_dir + '/' + i)
        s = midi2beat(mid, 'S', event2word)  # 1 means track
        a = midi2beat(mid, 'A', event2word)
        t = midi2beat(mid, 'T', event2word)
        b = midi2beat(mid, 'B', event2word)
        # merge SATB in same t as [S0, A0, T0, B0, ..., Sn, An, Tn, Bn]
        assert len(s) == len(a) == len(t) == len(b)
        beat = []
        for j in range(len(s)):
            beat.append(s[j])
            beat.append(a[j])
            beat.append(t[j])
            beat.append(b[j])
        beats.append(np.array(beat))
    merge_seqs = [(n, c, b) for n, c, b in zip(pitches, chord, beats)]
    return merge_seqs


def process_midi_track(mid, track_num):
    track = mid.tracks[track_num]
    pitch = []
    time = 256
    num = 3
    while num < len(track) - 2:
        if hasattr(track[num], 'note'):
            # note_on time=0:
            if track[num].dict()['type'] == 'note_on' and track[num].dict()['time'] == 0:
                for i in range(int(track[num+1].time/time)):
                    pitch.append(track[num].note)

            # note_on time!=0:
            if track[num].dict()['type'] == 'note_on' and track[num].dict()['time'] != 0:
                # add stop note
                for i in range(int(track[num].time/time)):
                    pitch.append(128)
                # add next note
                for i in range(int(track[num+1].time/time)):
                    pitch.append(track[num].note)
        num += 2
    return pitch


def midi2beat(mid, track_type, event2word):
    '''
    input: midi track and num of track to choose
    output: beat
    extract beat information from input:
    stop: no note
    start: the note on
    hold: after note on
    '''
    if track_type.islower():
        track_type.upper()
    track_dict = {'S': 1, 'A': 2, 'T': 3, 'B': 4}
    track = mid.tracks[track_dict[track_type]]
    beat = []
    time = 256
    num = 3
    while num < len(track) - 2:
        if hasattr(track[num], 'note'):
            # note_on time=0,
            if track[num].dict()['type'] == 'note_on' and track[num].dict()['time'] == 0:
                beat.append(event2word[track_type + '_start'])
                for i in range(int(track[num+1].time/time) - 1):
                    beat.append(event2word[track_type + '_hold'])

            # note_on time!=0:
            if track[num].dict()['type'] == 'note_on' and track[num].dict()['time'] != 0:
                # add stop note
                for i in range(int(track[num].time/time)):
                    beat.append(event2word[track_type + '_stop'])
                # add next note
                beat.append(event2word[track_type + '_start'])
                for i in range(int(track[num + 1].time / time) - 1):
                    beat.append(event2word[track_type + '_hold'])
        num += 2
    return beat


def process_jsb_by_bars(root, word2event):
    print("processing JSF data...")
    data_ori = np.load(root, allow_pickle=True, encoding='latin1')
    train_data = np.append(data_ori['valid'], data_ori['train'])
    data = np.append(train_data, data_ori['test'])
    # build dataset:
    seqs_bar = []
    seqs_bar_event = []
    for d in data:
        seqs_bar.append(d.transpose().reshape(-1).astype('int16'))

    return seqs_bar, seqs_bar_event


def split_dataset(seqs, output_dir, test_p=0.1):
    """
    split train, val, test
    """
    train_dir = os.path.join(output_dir, "train")
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(output_dir, "val")
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(val_dir, exist_ok=True)
    test_dir = os.path.join(output_dir, "test")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)

    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(seqs))
    test_set_size = int(len(seqs) * test_p)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    print("split dataset into train and test...")
    # put seqs to pkl
    for i in tqdm.tqdm(test_indices):
        with open(os.path.join(output_dir, 'test', str(i) + '.pkl'), 'wb') as f:
            pickle.dump(seqs[i], f)
    print("processing done:", os.path.join(output_dir, 'test'))
    for i in tqdm.tqdm(train_indices):
        with open(os.path.join(output_dir, 'train', str(i) + '.pkl'), 'wb') as f:
            pickle.dump(seqs[i], f)
    print("processing done:", os.path.join(output_dir, 'train'))


def parse_args():
    """

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default='./JSF_dataset/js-fakes-16thSeparated.npz',
                        help='the JSF dataset location')
    parser.add_argument("--midi_dir", type=str, default='./JSF_dataset/midi', help='the midi of dataset')
    parser.add_argument("--output_dir", type=str, default='./dataset/JSF_SATB',
                        help='Output folder')
    parser.add_argument("--data_amplification", default=True, help='up and low the melody to amplify the data')
    parser.add_argument("--data_reversal", default=True, help='reverse the melody')
    parser.add_argument("--arrangement", default=0, type=int,
                        help='0 means arrange by notes:CSATB, 1 means arrange by parts:CS,CA,CT,CB')
    return parser.parse_args()


def main():
    """

    """
    args = parse_args()
    root = args.root
    output_dir = args.output_dir
    event2word = build_event2word()
    word2event = build_word2event(event2word)
    print('Preprocessing files and saving to', output_dir)
    # seqs, seqs_event = process(root, args, word2event)
    # seqs, seqs_event = process_SATB(root, word2event, event2word)
    # seqs = process_midi_SATB(args.midi_dir, root, word2event, event2word)
    seqs = process_midi_CBSATB(args.midi_dir, root, word2event, event2word)
    split_dataset(seqs, output_dir)


if __name__ == '__main__':
    main()
