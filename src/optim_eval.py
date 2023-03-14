# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
below code is adopted from https://github.com/fatchord/WaveRNN
"""

import time
from typing import List
from pathlib import Path
import torch
from utils.text.symbols import symbols
from models.fatchord_version import WaveRNN
from models.tacotron import Tacotron
from utils import hparams as hp


def ipex_optimization(t_model, v_model, i_flag):
    """

    Args:
        t_model: Tacotron model
        v_model: WaveRNN model
        i_flag: Intel flag to enable ipex

    Returns: optimized models if intel ipex enabled

    """
    t_model.eval()
    v_model.eval()
    if i_flag:
        import intel_extension_for_pytorch as ipex
        t_model = ipex.optimize(t_model)
        v_model = ipex.optimize(v_model)
        print('\nINTEL IPEX Optimizations Enabled.\n')
    return t_model, v_model


def load_model(model_weights, typ, dv, return_model=False):
    """

    Args:
        return_model: Returns without weights
        model_weights: weights to load the model
        typ: model type (Tacotron / WaveRNN)
        dv: device

    Returns: loaded model

    """

    if typ == 'w':

        print('\nInitialising WaveRNN Model...\n')

        # Instantiate WaveRNN Model
        model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                        fc_dims=hp.voc_fc_dims,
                        bits=hp.bits,
                        pad=hp.voc_pad,
                        upsample_factors=hp.voc_upsample_factors,
                        feat_dims=hp.num_mels,
                        compute_dims=hp.voc_compute_dims,
                        res_out_dims=hp.voc_res_out_dims,
                        res_blocks=hp.voc_res_blocks,
                        hop_length=hp.hop_length,
                        sample_rate=hp.sample_rate,
                        mode='MOL').to(dv)

        if not return_model:
            model.load(model_weights)

    else:

        print('\nInitialising Tacotron Model...\n')

        # Instantiate Tacotron Model
        model = Tacotron(embed_dims=hp.tts_embed_dims,
                         num_chars=len(symbols),
                         encoder_dims=hp.tts_encoder_dims,
                         decoder_dims=hp.tts_decoder_dims,
                         n_mels=hp.num_mels,
                         fft_bins=hp.num_mels,
                         postnet_dims=hp.tts_postnet_dims,
                         encoder_K=hp.tts_encoder_K,
                         lstm_dims=hp.tts_lstm_dims,
                         postnet_K=hp.tts_postnet_K,
                         num_highways=hp.tts_num_highways,
                         dropout=hp.tts_dropout,
                         stop_threshold=hp.tts_stop_threshold).to(dv)

        if not return_model:
            model.load(model_weights)

    return model


def run_inference(text, taco_model, wave_model, batch, wrm_up, sv_path):
    """
        Performs the inference by taking input text and then give speech
    """

    sav_path = None
    tts = taco_model.get_step() // 1000
    inference_time = time.time()
    voc_time = 0
    tac_time = 0
    for i, x in enumerate(text, 1):

        if not wrm_up:

            tac_time = time.time()

        _, m, _ = taco_model.generate(x)

        if not wrm_up:
            print("\nTime taken by Tacotron model for inference is ", (time.time() - tac_time))
            voc_time = time.time()

        if batch:
            sav_path = f'{sv_path}/__input_batched{str(batch)}_{tts}k_{len(text)}_{i}.wav'

        else:
            sav_path = f'{sv_path}/__input_{"un" + str(batch)}__{tts}k_{len(text)}_{i}.wav'

        m = torch.tensor(m).unsqueeze(0)
        m = (m + 4) / 8
        # import pdb; pdb.set_trace()
        wave_model.generate(m, sav_path, batch, hp.voc_target, hp.voc_overlap, hp.mu_law)
        if not wrm_up:
            print("\nTime taken by WaveRNN model for inference is ", (time.time() - voc_time))

    if not wrm_up:
        print("\nTotal time for inference is ", (time.time() - inference_time))

    return sav_path


def levenshtein(a: List, b: List) -> int:
    """Calculates the Levenshtein distance between a and b.
    """
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


def word_error_rate(hypotheses: List[str], references: List[str]):
    """
    Computes Average Word Error rate between two texts represented as
    corresponding lists of string. Hypotheses and references must have same length.
    Args:
        hypotheses: list of hypotheses / predictions
        references: list of references / ground truth
    """
    scores = 0
    words = 0
    if len(hypotheses) != len(references):
        raise ValueError("In word error rate calculation, hypotheses and reference"
                         " lists must have the same number of elements. But I got:"
                         "{0} and {1} correspondingly".format(len(hypotheses), len(references)))
    for h, r in zip(hypotheses, references):
        h_list = h.split()
        r_list = r.split()
        words += len(r_list)
        scores += levenshtein(h_list, r_list)
    if words != 0:
        wer = (1.0 * scores) / words
    else:
        wer = float('inf')
    return wer, scores, words


def gen_testset(model: WaveRNN, test_set, samples, batched, target, overlap, save_path: Path):
    k = model.get_step() // 1000

    for i, (m, x) in enumerate(test_set, 1):

        if i > samples:
            break

        print('\n| Generating: %i/%i' % (i, samples))

        x = x[0].numpy()

        bits = 16 if hp.voc_mode == 'MOL' else hp.bits

        if hp.mu_law and hp.voc_mode != 'MOL':
            x = decode_mu_law(x, 2 ** bits, from_labels=True)
        else:
            x = label_2_float(x, bits)

        save_wav(x, save_path / f'{k}k_steps_{i}_target.wav')

        batch_str = f'gen_batched_target{target}_overlap{overlap}' if batched else 'gen_NOT_BATCHED'
        save_str = str(save_path / f'{k}k_steps_{i}_{batch_str}.wav')

        _ = model.generate(m, save_str, batched, target, overlap, hp.mu_law)
