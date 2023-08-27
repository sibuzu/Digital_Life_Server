import wave

import numpy as np
import pyaudio
import os

from TTS.TTService import TTService, TTSEngin
import TTS.vits.utils as utils
from TTS.vits.models import SynthesizerTrn
from TTS.vits.text import create_symbol_id_map

text_jp = '[JA]その通りです。車は頻繁に使うものですので、安全性も大切にしなければなりません[JA]'

xpath = '../vits-models/pretrained_models'
models = os.listdir(xpath)
cfg = 'TTS/models/momoi.json'

tts_service = TTService()
for m in models:
    mfile = f"{xpath}/{m}/{m}.pth"
    if not os.path.isfile(mfile):
        continue

    tts = TTSEngin()
    tts.hps = utils.get_hparams_from_file(cfg)
    
    tts.symbol_to_id, tts.id_to_symbol = create_symbol_id_map(tts.hps.symbols)

    tts.engine = SynthesizerTrn(
        len(tts.hps.symbols),
        tts.hps.data.filter_length // 2 + 1,
        tts.hps.train.segment_size // tts.hps.data.hop_length,
        **tts.hps.model).cuda()
    _ = tts.engine.eval()
    _ = utils.load_checkpoint(mfile, tts.engine, None)
 
    print(f"Actor: {m}")
    audio = tts.read(text_jp)
    data = audio.astype(np.float32).tobytes()
    # Set the output file name
    output_file = f"../vits-models/jp_voices/{m}.wav"

    # Set the audio properties
    num_channels = 1
    sample_width = 2  # Assuming 16-bit audio
    frame_rate = tts.hps.data.sampling_rate

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
