import logging
import time

# ASR
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

class ASRService():
    def __init__(self):
        logging.info('Initializing ASR Service...')
        # PARAFORMER
        models = [
            '../models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
            '../models/speech_paraformer_asr-en-16k-vocab4199-pytorch',
            '../models/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-offline',
        ]
        self.pipes = [pipeline(task=Tasks.auto_speech_recognition, model=model) 
                      for model in models]

    def infer(self, wav_path, voice_lang):
        stime = time.time()
        assert voice_lang >= 0 and voice_lang < len(self.pipes), f'Invalud voice lang {voice_lang}'
        result = self.pipes[voice_lang](audio_in=wav_path)
        logging.info('ASR Result: %s. time used %.2f.' % (result, time.time() - stime))
        return str(result)

if __name__ == '__main__':
    config_path = 'ASR/resources/config.yaml'

    service = ASRService(config_path)

    # print(wav_path)
    wav_path = 'ASR/test_wavs/0478_00017.wav'
    result = service.infer(wav_path)
    print(result)

    wav_path = 'ASR/test_wavs/asr_example_zh.wav'
    result = service.infer(wav_path)
    print(result)