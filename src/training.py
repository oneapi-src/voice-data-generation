# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
Train Tacotron model
below code is adopted from https://github.com/fatchord/WaveRNN
"""
# pylint: disable=C0103,C0301,C0413,E0401,W0614,C0412,W0401,W0613,R0914,R0913,R0915

import time
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from utils import hparams as hp
from utils.display import *
from utils.dataset import get_tts_datasets, get_vocoder_datasets
from utils.text.symbols import symbols
from utils.distribution import discretized_mix_logistic_loss
from utils.paths import Paths
from models.tacotron import Tacotron
from models.fatchord_version import WaveRNN
from utils import data_parallel_workaround
from utils.optim_eval import gen_testset
from utils.checkpoints import save_checkpoint, restore_checkpoint


def np_now(x: torch.Tensor):
    """
        converting torch Tensor to numpy array
    """
    return x.detach().cpu().numpy()


def main():
    """
        Main method
    """
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train Tacotron TTS & WaveRNN Voc')
    parser.add_argument('--force_gta', '-g', action='store_true', help='Force the model to create GTA features')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA '
                                                                       'capable environment')
    parser.add_argument('--lr', '-l', type=float, help='[float] override hparams.py learning rate')
    parser.add_argument('--batch_size', '-b', type=int, help='[int] override hparams.py batch size')
    parser.add_argument('--hp_file', metavar='FILE', default='src/utils/hparams.py',
                        help='The file to use for the hyper parameters')
    parser.add_argument('--epochs', '-e', type=int, default=None,
                        help='[int] number of epochs for training')


    args = parser.parse_args()

    hp.configure(args.hp_file)  # Load hparams from file
    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    if args.lr is None:
        args.lr = hp.voc_lr
    if args.batch_size is None:
        args.batch_size = hp.voc_batch_size

    batch_size = args.batch_size
    lr = args.lr
    train_gta = args.force_gta
    epochs = args.epochs

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        for session in hp.tts_schedule:
            _, _, _, batch_size = session
            if batch_size % torch.cuda.device_count() != 0:
                raise ValueError('`batch_size` must be evenly divisible by n_gpus!')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    # Instantiate Tacotron Model
    print('\nInitialising Tacotron Model...\n')
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
                     stop_threshold=hp.tts_stop_threshold).to(device)

    optimizer = optim.Adam(model.parameters())

    restore_checkpoint('tts', paths, model, optimizer, create_if_missing=True)

    start = time.time()
    for _, session in enumerate(hp.tts_schedule):
        current_step = model.get_step()

        r, lr, max_step, batch_size = session

        training_steps = max_step - current_step
        model.r = r
        simple_table([(f'Steps with r={r}', str(training_steps // 1000) + 'k Steps'),
                      ('Batch Size', batch_size),
                      ('Learning Rate', lr),
                      ('Outputs/Step (r)', model.r)])

        train_set, attn_example = get_tts_datasets(paths.data, batch_size, r)

        tts_train_loop(paths, model, optimizer, train_set, lr, training_steps, attn_example, epochs)

    print("Total training time is ", (time.time() - start))
    print('Training Tacotron model is Completed.')


    print('Creating Ground Truth Aligned Dataset...\n')

    train_set, attn_example = get_tts_datasets(paths.data, 32, model.r)
    create_gta_features(model, train_set, paths.gta)

    print('\n\nWe can now train WaveRNN on GTA features\n')

    print('\nInitialising WaveRNN Model...\n')

    # Instantiate WaveRNN Model
    voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
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
                        mode=hp.voc_mode).to(device)

    # Check to make sure the hop length is correctly factorised
    assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length

    optimizer = optim.Adam(voc_model.parameters())

    restore_checkpoint('voc', paths, voc_model, optimizer, create_if_missing=True)

    train_set, test_set = get_vocoder_datasets(paths.data, batch_size, train_gta)

    total_steps = hp.voc_total_steps

    simple_table([('Remaining', str((total_steps - voc_model.get_step()) // 1000) + 'k Steps'),
                  ('Batch Size', batch_size),
                  ('LR', lr),
                  ('Sequence Len', hp.voc_seq_len),
                  ('GTA Train', train_gta)])

    loss_func = F.cross_entropy if voc_model.mode == 'RAW' else discretized_mix_logistic_loss

    start = time.time()
    voc_train_loop(paths, voc_model, loss_func, optimizer, train_set, test_set, lr, total_steps, epochs)

    print("Total training time to train WaveRNN model is ", (time.time() - start))
    print('\nTraining Completed for both Tacotron and WaveRNN models.')


def tts_train_loop(paths: Paths, model: Tacotron, optimizer, train_set, lr, train_steps, attn_example, eps=None):
    """
        Training Tacotron model
    """
    device = next(model.parameters()).device  # use same device as model parameters

    for g in optimizer.param_groups:
        g['lr'] = lr

    total_iters = len(train_set)

    epochs = eps if eps else train_steps // total_iters + 1


    msg = None
    start = time.time()
    for e in range(1, epochs + 1):

        running_loss = 0
        strt = time.time()
        # Performs 1 iteration for every input string
        for i, (x, m, ids, _) in enumerate(train_set, 1):

            x, m = x.to(device), m.to(device)

            # Parallelize model onto GPUS using workaround due to python bug
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                m1_hat, m2_hat, attention = data_parallel_workaround(model, x, m)
            else:
                m1_hat, m2_hat, attention = model(x, m)

            m1_loss = F.l1_loss(m1_hat, m)
            m2_loss = F.l1_loss(m2_hat, m)

            loss = m1_loss + m2_loss

            optimizer.zero_grad()
            loss.backward()
            if hp.tts_clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.tts_clip_grad_norm)
                if np.isnan(grad_norm):
                    print('grad_norm was NaN!')

            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / i

            speed = i / (time.time() - strt)

            step = model.get_step()
            k = step // 1000

            if step % hp.tts_checkpoint_every == 0:
                ckpt_name = f'taco_step{k}K'
                save_checkpoint('tts', paths, model, optimizer,
                                name=ckpt_name, is_silent=True)

            if attn_example in ids:
                idx = ids.index(attn_example)
                save_attention(np_now(attention[idx][:, :160]), paths.tts_attention / f'{step}')
                save_spectrogram(np_now(m2_hat[idx]), paths.tts_mel_plot / f'{step}', 600)

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: ' \
                  f'{avg_loss:#.4} | {speed:#.2} steps/s | Step: {k}k | '
            stream(msg)

        save_checkpoint('tts', paths, model, optimizer, is_silent=True)
        model.log(paths.tts_log, msg)
        print(' ')
    print("Total Training time for Tacotron Model is ", (time.time() - start))


def create_gta_features(model: Tacotron, train_set, save_path: Path):
    """
        Creating Ground Truth aligned features incase we use it for training later
    """
    device = next(model.parameters()).device  # use same device as model parameters

    iters = len(train_set)

    for i, (x, mels, ids, mel_lens) in enumerate(train_set, 1):

        x, mels = x.to(device), mels.to(device)

        with torch.no_grad():
            _, gta, _ = model(x, mels)

        gta = gta.cpu().numpy()

        for j, item_id in enumerate(ids):
            mel = gta[j][:, :mel_lens[j]]
            mel = (mel + 4) / 8
            np.save(save_path / f'{item_id}.npy', mel, allow_pickle=False)

        bar1 = progbar(i, iters)
        msg = f'{bar1} {i}/{iters} Batches '
        stream(msg)


def voc_train_loop(paths: Paths, model: WaveRNN, loss_func, optimizer, train_set, test_set, lr, total_steps, eps=None):
    """
        Training WaveRNN model
    """
    # Use same device as model parameters
    device = next(model.parameters()).device

    for g in optimizer.param_groups:
        g['lr'] = lr

    total_iters = len(train_set)

    epochs = eps if eps else (total_steps - model.get_step()) // total_iters + 1
    for e in range(1, epochs + 1):

        start = time.time()
        running_loss = 0.

        for i, (x, y, m) in enumerate(train_set, 1):
            x, m, y = x.to(device), m.to(device), y.to(device)

            # Parallelize model onto GPUS using workaround due to python bug
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                y_hat = data_parallel_workaround(model, x, m)
            else:
                y_hat = model(x, m)

            if model.mode == 'RAW':
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)

            elif model.mode == 'MOL':
                y = y.float()

            y = y.unsqueeze(-1)

            loss = loss_func(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            if hp.voc_clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.voc_clip_grad_norm)
                if np.isnan(grad_norm):
                    print('grad_norm was NaN!')
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / i

            speed = i / (time.time() - start)

            step = model.get_step()
            k = step // 1000

            if step % hp.voc_checkpoint_every == 0:
                gen_testset(model, test_set, hp.voc_gen_at_checkpoint, hp.voc_gen_batched,
                            hp.voc_target, hp.voc_overlap, paths.voc_output)
                ckpt_name = f'wave_step{k}K'
                save_checkpoint('voc', paths, model, optimizer,
                                name=ckpt_name, is_silent=True)

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: ' \
                  f'{avg_loss:.4f} | {speed:.1f} steps/s | Step: {k}k | '
            stream(msg)

        # Must save latest optimizer state to ensure that resuming training
        # doesn't produce artifacts
        save_checkpoint('voc', paths, model, optimizer, is_silent=True)
        model.log(paths.voc_log, msg)
        print(' ')


if __name__ == "__main__":
    main()
