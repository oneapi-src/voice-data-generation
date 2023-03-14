# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
Evaluation code
below code is adopted from https://github.com/fatchord/WaveRNN
"""
# pylint: disable=C0103,C0301,C0413,E0401,R0914,R0915

# !pip install SpeechRecognition pydub

import argparse
import os
import torch
import speech_recognition as sr
import soundfile
from utils.optim_eval import word_error_rate, ipex_optimization, load_model, run_inference
from utils import hparams as hp
from utils.text import text_to_sequence
from utils.display import simple_table


def process_evaluation_epoch(global_vars: dict):
    """
    Processes results from each worker at the end of evaluation and combine to final result
    Args:
        global_vars: dictionary containing information of entire evaluation
    Return:
        wer: final word error rate
        loss: final loss
    """
    hypotheses = global_vars['predictions']
    references = global_vars['transcripts']

    # wer, scores, num_words
    wer, _, _ = word_error_rate(
        hypotheses=hypotheses, references=references)
    return wer


def main():
    """
        Main function
    """
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--input_text', '-i', type=str, default=None,
                        help='[string] Type in something here and TTS will generate it!')
    parser.add_argument('--batched', '-b', dest='batched', action='store_true', default=False,
                        help='Fast Batched Generation (lower quality)')
    parser.add_argument('--unbatched', '-u', dest='batched', action='store_false',
                        help='Slower Unbatched Generation (better quality)')
    parser.add_argument('--force_cpu', '-c', action='store_true',
                        help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py',
                        help='The file to use for the hyper parameters')
    parser.add_argument('-ipx', '--intel', type=int, required=False, default=0,
                        help='use 1 for enabling intel pytorch optimizations, default is 0')
    parser.add_argument('--save_path', type=str, default='saved_audio/evaluation',
                        help='[string/path] where to store the speech files generated for the input text, '
                             'default saved_audio folder')
    parser.add_argument('--voc_weights', type=str, default='pretrained/voc_weights/latest_weights.pyt',
                        help='[string/path] Load in different WaveRNN weights')
    parser.add_argument('--tts_weights', type=str, default='pretrained/tts_weights/latest_weights.pyt',
                        help='[string/path] Load in different Tacotron weights')
    args = parser.parse_args()

    hp.configure(args.hp_file)  # Load hparams from file

    # parser.set_defaults(batched=False)
    parser.set_defaults(input_text=None)

    batched = args.batched
    input_text = args.input_text
    intel_flag = args.intel
    tts_weights = args.tts_weights
    voc_weights = args.voc_weights
    save_path = args.save_path
    # creating the save path directory to store the output generated audio file if it does not exist
    os.makedirs(save_path, exist_ok=True)

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    if not (input_text.endswith(".txt") or input_text.endswith(".csv")):

        inputs = [text_to_sequence(input_text.strip(), hp.tts_cleaner_names)]
        inp_txt = input_text
    else:
        with open(input_text) as f:
            inputs = []
            inp_txt = []
            cnt = 0
            for line in f:
                split = line.split(',')
                sentence = split[-1][:-1]
                # adding "hi " here because the speech to text conversion sometimes misses the 1st word
                sentence = "hi " + sentence
                if cnt > 0:
                    inp_txt.append(sentence)
                    inputs.append(text_to_sequence(sentence.strip(), hp.tts_cleaner_names))
                cnt += 1

    voc_model = load_model(voc_weights, 'w', device)
    tts_model = load_model(tts_weights, 't', device)

    tts_model, voc_model = ipex_optimization(tts_model, voc_model, intel_flag)

    voc_k = voc_model.get_step() // 1000
    tts_k = tts_model.get_step() // 1000

    r = tts_model.r

    simple_table([('WaveRNN', str(voc_k) + 'k'),
                  (f'Tacotron(r={r})', str(tts_k) + 'k'),
                  ('Generation Mode', 'Batched' if batched else 'Unbatched'),
                  ('Target Samples', 11_000 if batched else 'N/A'),
                  ('Overlap Samples', 550 if batched else 'N/A')])

    wer = 0.
    itr = 1
    for i, x in enumerate(inputs, 1):

        print(f'\n\nGenerating speech for the input passed line {i}...\n')

        _, m, _ = tts_model.generate(x)

        if batched:
            sav_path = f'{save_path}/__input_batched{str(batched)}_{tts_k}k_{len(inputs)}_{i}.wav'

        else:
            sav_path = f'{save_path}/__input_{"un" + str(batched)}__{tts_k}k_{len(inputs)}_{i}.wav'

        m = torch.tensor(m).unsqueeze(0)
        m = (m + 4) / 8
        voc_model.generate(m, sav_path, batched, hp.voc_target, hp.voc_overlap, hp.mu_law)

        data, samplerate = soundfile.read(sav_path)
        soundfile.write('new.wav', data, samplerate, subtype='PCM_16')
        filename = 'new.wav'

        # initialize the recognizer
        r = sr.Recognizer()
        with sr.AudioFile(filename) as source:
            # listen for the data (load audio to memory)
            audio_data = r.record(source)
            # recognize (convert from speech to text)
            text_pre = r.recognize_google(audio_data)
            text_pre = "".join(letter for letter in text_pre if letter.isalnum() or letter == " ")
        # making the input text case-insensitive
        if not (input_text.endswith(".txt") or input_text.endswith(".csv")):

            text_gt = "".join(letter for letter in inp_txt if letter.isalnum() or letter == " ")

        else:
            text_gt = ''.join(letter for letter in inp_txt[i - 1] if letter.isalnum() or letter == " ")
            # dropping "hi " here because we added in speech gen
            text_gt = text_gt[2:]

        text_pre = text_pre.lower().strip()
        text_gt = text_gt.lower().strip()

        if len(text_gt) != len(text_pre):
            if len(text_gt) > len(text_pre):
                for _ in range(len(text_gt) - len(text_pre)):
                    text_pre += " "
            else:
                text_pre = text_pre[:len(text_gt)]

        references = [text_gt]
        hypotheses = [text_pre]

        d = dict(predictions=hypotheses,
                 transcripts=references)
        wer += process_evaluation_epoch(d)
        itr += 1
        print("Input sentence passed::\n", text_gt)
        print("Predicted sentence of the model::\n", text_pre)

    wer /= itr - 1
    print("Number of sentences:", itr - 1)
    print("\nAverage Word Error Rate: {:}%, accuracy={:}%".format(wer * 100, (1 - wer) * 100), "\n")


if __name__ == '__main__':
    main()
