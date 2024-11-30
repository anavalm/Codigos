import pyaudio
import numpy as np
import librosa
import math
import os
import wave
from scipy.spatial.distance import pdist

# Parámetros de grabación y procesamiento
RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 6
RESPEAKER_WIDTH = 2
RESPEAKER_INDEX = 1
CHUNK = 1024
BUFFER_SECONDS = 4
UPDATE_INTERVAL = 1
SEGMENT_DURATION = 10  # Duración de cada segmento en segundos
OUTPUT_FOLDER_AUDIO = "audio_output"
OUTPUT_FOLDER_GCC_PHAT = "gcc_phat_output"
OUTPUT_FOLDER_SPECTROGRAM = "spectrogram_output"
SELECTED_CHANNEL = 2  # Canal a guardar (0-indexed) microfono superior del casco

# Parámetros FFT
hop_length = math.ceil(0.001 * RESPEAKER_RATE)  # 1 ms
nfft_lenght = math.ceil(0.05 * RESPEAKER_RATE)  # 50 ms

# Posiciones de micrófonos
mic_pos = np.array([[-0.175, 0, 0], [0, 0.01, 0.1], [0, -0.01, 0.1], [0.175, 0, 0]])

# Crear carpetas de salida
os.makedirs(OUTPUT_FOLDER_AUDIO, exist_ok=True)
os.makedirs(OUTPUT_FOLDER_GCC_PHAT, exist_ok=True)
os.makedirs(OUTPUT_FOLDER_SPECTROGRAM, exist_ok=True)

def max_tau(mic_pos, fs, margen_tau):
    distances = pdist(mic_pos)
    max_delays = max(np.round(distances / 343.0 * fs).astype(int)) + margen_tau
    return math.ceil(max_delays / hop_length)

def gcc_phat_pairwise(audio, delay_max_muestras):
    """
    Calcula GCC-PHAT y retorna los espectrogramas y GCC-PHAT para cada par de micrófonos.
    """
    num_channels = audio.shape[0]
    gcc_phat_pairs = []
    spectrogram_pairs = []

    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            audio_pair = np.vstack((audio[i, :], audio[j, :]))
            stft_mic1 = librosa.stft(audio_pair[0, :], n_fft=nfft_lenght, hop_length=hop_length)
            spectrogram_pairs.append(abs(stft_mic1))
            stft_mic2 = librosa.stft(audio_pair[1, :], n_fft=nfft_lenght, hop_length=hop_length)
            spectrogram_pairs.append(abs(stft_mic2))
            #print(stft_mic2.shape)

            # Calcular GCC-PHAT
            cross_corr = stft_mic1 * np.conj(stft_mic2)
            gcc_phat = cross_corr / np.abs(cross_corr)
            gcc_phat = np.fft.irfft(gcc_phat)

            gcc_phat = np.concatenate(
                [gcc_phat[:, -delay_max_muestras:], gcc_phat[:, :delay_max_muestras]],
                axis=-1,
            )
            gcc_phat_pairs.append(gcc_phat)

    return np.array(gcc_phat_pairs), spectrogram_pairs

def save_audio(frames, segment_index):
    """Guardar el audio grabado en un archivo WAV."""
    output_filename = os.path.join(OUTPUT_FOLDER_AUDIO, f"channel_{SELECTED_CHANNEL}_segment_{segment_index}.wav")
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16 bits = 2 bytes
        wf.setframerate(RESPEAKER_RATE)
        wf.writeframes(b''.join(frames))
    print(f"Audio guardado: {output_filename}")

def save_gcc_phat(gcc_phat_data, segment_index):
    """Guardar los resultados de GCC-PHAT en un archivo NPY."""
    output_filename = os.path.join(OUTPUT_FOLDER_GCC_PHAT, f"gcc_phat_segment_{segment_index}.npy")
    np.save(output_filename, gcc_phat_data)
    print(f"GCC-PHAT guardado: {output_filename}")

def save_spectrograms(spectrogram_data, segment_index):
    """Guardar los espectrogramas como lista en un archivo NPY."""
    output_filename = os.path.join(OUTPUT_FOLDER_SPECTROGRAM, f"spectrogram_segment_{segment_index}.npy")
    np.save(output_filename, spectrogram_data, allow_pickle=True)
    print(f"Espectrogramas guardados: {output_filename}")

def real_time_processing():
    p = pyaudio.PyAudio()
    stream = p.open(
        rate=RESPEAKER_RATE,
        format=p.get_format_from_width(RESPEAKER_WIDTH),
        channels=RESPEAKER_CHANNELS,
        input=True,
        input_device_index=RESPEAKER_INDEX,
        frames_per_buffer=CHUNK
    )

    print("* Procesando en tiempo real... Presiona 'Ctrl+C' para detener.")

    ventana_muestras = BUFFER_SECONDS * RESPEAKER_RATE
    paso_muestras = UPDATE_INTERVAL * RESPEAKER_RATE
    print(paso_muestras)
    buffer = np.zeros((4, ventana_muestras))  # Usamos los canales 2, 3, 4, 5
    delay_max = max_tau(mic_pos, RESPEAKER_RATE, margen_tau=math.ceil(0.1 / 343.0 * RESPEAKER_RATE))

    segment_index = 0
    frames_audio = []
    gcc_phat_data = []
    spectrogram_data = []

    try:
        while True:
            new_frames = []
            for _ in range(int(RESPEAKER_RATE / CHUNK * UPDATE_INTERVAL)):
                data = stream.read(CHUNK)
                #print(len(data))
                frames_audio.append(
                    np.frombuffer(data, dtype=np.int16)[SELECTED_CHANNEL::RESPEAKER_CHANNELS].tobytes()
                )
                
                new_data = np.frombuffer(data, dtype=np.int16).reshape(RESPEAKER_CHANNELS, -1)
                selected_channels = new_data[1:5, :]  # Canales 2, 3, 4, 5
                new_frames.append(selected_channels)

            new_frames = np.hstack(new_frames)
            buffer = np.hstack((buffer[:, new_frames.shape[1]:], new_frames))

            # Procesar buffer con GCC-PHAT y obtener espectrogramas
            gcc_phat_result, spectrogram_result = gcc_phat_pairwise(buffer, delay_max)
            print(buffer.shape,new_frames.shape)
            gcc_phat_data.append(gcc_phat_result)
            spectrogram_data.extend(spectrogram_result)

            # Guardar resultados cada 10 segundos
            if len(frames_audio) >= int(RESPEAKER_RATE * SEGMENT_DURATION / CHUNK):
                save_audio(frames_audio, segment_index)
                save_gcc_phat(np.array(gcc_phat_data), segment_index)
                save_spectrograms(spectrogram_data, segment_index)
                frames_audio = []
                gcc_phat_data = []
                spectrogram_data = []
                segment_index += 1

    except KeyboardInterrupt:
        print("Procesamiento detenido por el usuario.")
        if frames_audio:
            save_audio(frames_audio, segment_index)
            save_gcc_phat(np.array(gcc_phat_data), segment_index)
            save_spectrograms(spectrogram_data, segment_index)

    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    real_time_processing()
