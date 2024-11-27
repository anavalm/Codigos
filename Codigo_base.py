import librosa
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist



def gcc_phat_pairwise(audio, delay_max_samples):
    """Calculates GCC-PHAT between all channel pairs"""
    num_channels = audio.shape[0]
    gcc_phat_pairs = []

    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            # Pair signals
            audio_pair = np.vstack((audio[i, :], audio[j, :]))

            # STFT
            stft_mic1 = librosa.stft(audio_pair[0, :], n_fft=nfft_length, hop_length=hop_length)
            stft_mic2 = librosa.stft(audio_pair[1, :], n_fft=nfft_length, hop_length=hop_length)

            # GCC-PHAT calculation
            cross_corr = stft_mic1 * np.conj(stft_mic2)
            gcc_phat = cross_corr / np.abs(cross_corr)
            gcc_phat = np.fft.irfft(gcc_phat)

            # Windowing GCC-PHAT around max delay
            gcc_phat = np.concatenate(
                [gcc_phat[:, -delay_max_samples:], gcc_phat[:, :delay_max_samples]],
                axis=-1,
            )

            gcc_phat_pairs.append(gcc_phat)

    return np.array(gcc_phat_pairs)

def max_tau(mic_positions, fs, margin_tau):
    """Calculate max delay in samples with added margin"""
    distances = pdist(mic_positions)
    max_delays = max(np.round(distances / 343.0 * fs).astype(int)) + margin_tau
    max_delays = math.ceil(max_delays / hop_length)
    return max_delays


if __name__ == "__main__":
    audio, sr = librosa.load('Audio_6_canales.wav', sr=None, mono=False)
    audio = audio[1:5,:]

    # User-defined microphone positions
    mic_pos = np.array([[-0.175, 0, 0.1], [0, 0.01, 0.1], [0, -0.01, 0.1], [-0.175, 0, 0.1]])
    margen_distancia = 0.5  # distance margin in meters
    margen_tau = math.ceil((margen_distancia / 343.0) * sr)
    

    hop_length = math.ceil(0.001 * sr)  # 1 ms hop length
    nfft_length = math.ceil(0.050 * sr)  # 50 ms FFT window

    delay_max = max_tau(mic_pos, sr, margen_tau)
    espectrogramas = gcc_phat_pairwise(audio, delay_max)
    print("La shape es:", np.shape(espectrogramas))
