import wave

import numpy as np
import pyaudio

from TTS.TTService import TTService

text_ch_1 = '旅行者，今天是星期四，能否威我五十'
text_en_1 = 'Travller, today is Thursday. How old are you?'
text_jp_1 = '[JA]その通りです。車は頻繁に使うものですので、安全性も大切にしなければなりません[JA]'
config_combo = [
        ("TTS/models/paimon6k.json", "TTS/models/paimon6k_390k.pth", text_ch_1),
        ("TTS/models/momoi.json", "TTS/models/momoi.pth", text_jp_1),
    ]
for cfg, model, text in config_combo:
    a = TTService(cfg, model, 'test', 1)
    p = pyaudio.PyAudio()
    audio = a.read(text)
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=a.hps.data.sampling_rate,
                    output=True
                    )
    data = audio.astype(np.float32).tobytes()
    stream.write(data)
    # Set the output file name
    output_file = "output.wav"

    # Set the audio properties
    num_channels = 1
    sample_width = 2  # Assuming 16-bit audio
    frame_rate = a.hps.data.sampling_rate

    # Convert audio data to 16-bit integers
    audio_int16 = (audio * np.iinfo(np.int16).max).astype(np.int16)

    # Open the output file in write mode
    with wave.open(output_file, 'wb') as wav_file:
        # Set the audio properties
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(frame_rate)

        # Write audio data to the file
        wav_file.writeframes(audio_int16.tobytes())