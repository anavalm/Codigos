import pyaudio
import wave
import os
import keyboard
import numpy as np
# Parámetros de grabación
RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 6  # Cambia según la configuración del dispositivo
RESPEAKER_WIDTH = 2
RESPEAKER_INDEX = 1  # Índice del dispositivo de entrada
CHUNK = 1024
SEGMENT_DURATION = 60  # Duración de cada segmento en segundos
OUTPUT_FOLDER = "audio_output"
SELECTED_CHANNEL = 1  # Cambiar para elegir el canal deseado (0-indexed)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(
        rate=RESPEAKER_RATE,
        format=p.get_format_from_width(RESPEAKER_WIDTH),
        channels=RESPEAKER_CHANNELS,
        input=True,
        input_device_index=RESPEAKER_INDEX,
        frames_per_buffer=CHUNK
    )

    print("* Grabando... Presiona 'q' para detener.")

    segment_index = 0

    try:
        while True:
            frames = []
            for _ in range(0, int(RESPEAKER_RATE / CHUNK * SEGMENT_DURATION)):
                data = stream.read(CHUNK)
                # Extraer datos del canal seleccionado
                channel_data = wave.open
                channel_data = np.frombuffer(data, dtype=np.int16)[SELECTED_CHANNEL::RESPEAKER_CHANNELS]
                frames.append(channel_data.tobytes())

            # Guardar el archivo de audio del canal seleccionado
            output_filename = os.path.join(OUTPUT_FOLDER, f"channel_{SELECTED_CHANNEL}_segment_{segment_index}.wav")
            with wave.open(output_filename, 'wb') as wf:
                wf.setnchannels(1)  # Archivo mono
                wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
                wf.setframerate(RESPEAKER_RATE)
                wf.writeframes(b''.join(frames))

            print(f"Segmento {segment_index} guardado: {output_filename}")
            segment_index += 1

    except KeyboardInterrupt:
        pass  # Para manejar un cierre manual seguro

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Grabación detenida.")

if __name__ == "__main__":
    record_audio()
