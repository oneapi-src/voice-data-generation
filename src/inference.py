# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
Inference code for trained models
below code is adopted from https://github.com/fatchord/WaveRNN
"""
# pylint: disable=C0103,C0301,C0413,E0401,E1101,C0415,

import time
import argparse
import os
import sys
import torch

from utils.optim_eval import ipex_optimization, load_model, run_inference
from utils import hparams as hp
from utils.text import text_to_sequence
from utils.display import simple_table
from utils.files import get_files


def main():
    """
        Main Function
    """
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--input_text', '-i', type=str, default=None,
                        help='[string/csv file] Type in something here and TTS will generate it!')
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
    parser.add_argument('--save_path', type=str, default='saved_audio',
                        help='[string/path] where to store the speech files generated for the input text, '
                             'default saved_audio folder')
    parser.add_argument('--voc_weights', type=str, default='pretrained/voc_weights/latest_weights.pyt',
                        help='[string/path] Load in different WaveRNN weights')
    parser.add_argument('--tts_weights', type=str, default='pretrained/tts_weights/latest_weights.pyt',
                        help='[string/path] Load in different Tacotron weights')
    args = parser.parse_args()

    hp.configure(args.hp_file)  # Load hparams from file

    parser.set_defaults(input_text=None)

    batched = args.batched
    input_text = args.input_text
    intel_flag = args.intel
    tts_weights = args.tts_weights
    voc_weights = args.voc_weights
    save_path = args.save_path

    # creating the save path directory to store the output generated audio file if it does not exist
    os.makedirs(save_path, exist_ok=True)

    if not (input_text.endswith(".txt") or input_text.endswith(".csv")):
        inputs = [text_to_sequence(input_text.strip(), hp.tts_cleaner_names)]
    else:
        with open(input_text) as f:

            inputs = []
            cnt = 0
            for line in f:
                split = line.split(',')
                sentence = split[-1][:-1].strip()
                if cnt > 0:
                    inputs.append(text_to_sequence(sentence.strip(), hp.tts_cleaner_names))
                cnt += 1

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

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

    print("\nWarming Up the models for inference.....")
    for _ in range(0, 10):
        wr_up = True
        warm_up_ip = "This is an input used for warming up our models"
        wr_up_ip = [text_to_sequence(warm_up_ip.strip(), hp.tts_cleaner_names)]
        run_inference(wr_up_ip, tts_model, voc_model, batched, wr_up, save_path)

    print('\nFinished warmup.\nGenerating speech for the input passed...\n')

    wr_up = False
    run_inference(inputs, tts_model, voc_model, batched, wr_up, save_path)

    print('\n\nDone.\n')


if __name__ == "__main__":
    main()
