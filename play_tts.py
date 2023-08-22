import wave

import numpy as np
import pyaudio

from TTS.TTService import TTService

### momoi.pth 
###   from https://huggingface.co/spaces/zomehwh/vits-models
### pretrained_ljs.pth
###   from https://drive.google.com/uc?id=1q86w74Ygw2hNzYP9cWkeClGT5X25PvBT
###   config ljs_base.json from https://github.com/jaywalnut310/vits/tree/main/configs
### pretrained_vctk.pth
###   from https://drive.google.com/uc?id=11aHOlhnxzjpdWDpsz1vFDCzbeEfoIxru
###   config vctk_base.json from https://github.com/jaywalnut310/vits/tree/main/configs

text_ch_1 = '旅行者，今天是星期四，能否威我五十'
text_en_1 = 'Travller, today is Thursday. How old are you?'
text_en_2 = 'A rainbow is a beautiful, colorful arch of light that can sometimes appear in the sky after it rains.'
text_jp_1 = '[JA]その通りです。車は頻繁に使うものですので、安全性も大切にしなければなりません[JA]'
config_combo = [
        (0, text_ch_1),
        (2, text_jp_1),
        (1, text_en_1),
        (1, text_en_2),
    ]

tts_service = TTService()
for actor, text in config_combo:
    a = tts_service.get_tts(actor)
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