# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""librosa-backed MelSpectrogram matching torchaudio's mel_scale="slaney",
norm="slaney" convention exactly (torchaudio's own slaney mode is designed
to reproduce librosa's default htk=False, norm="slaney" mel filterbank -
this is the well-documented interoperability target between the two
libraries, not a coincidence). STFT convention (hann window, center=True,
reflect padding) matches both libraries' shared defaults.
"""
import numpy as np
import librosa
import torch


class MelSpectrogram:
    def __init__(self, sample_rate, n_fft, win_length, hop_length, f_min, f_max,
                 n_mels, power=2.0, mel_scale="slaney", norm="slaney", **_ignored):
        if mel_scale != "slaney" or norm != "slaney":
            raise NotImplementedError("this shim only reproduces mel_scale=slaney, norm=slaney")
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.power = power
        self.mel_fb = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=f_min, fmax=f_max,
            htk=False, norm="slaney",
        ).astype(np.float64)

    def __call__(self, waveform):
        wf = waveform.detach().cpu().numpy().astype(np.float64)
        lead_shape = wf.shape[:-1]
        flat = wf.reshape(-1, wf.shape[-1])
        rows = []
        for row in flat:
            stft = librosa.stft(
                row, n_fft=self.n_fft, hop_length=self.hop_length,
                win_length=self.win_length, window="hann", center=True, pad_mode="reflect",
            )
            mag = np.abs(stft) ** self.power
            rows.append(self.mel_fb @ mag)
        out = np.stack(rows, axis=0).reshape(*lead_shape, rows[0].shape[0], rows[0].shape[1])
        return torch.from_numpy(out).to(waveform.dtype)
