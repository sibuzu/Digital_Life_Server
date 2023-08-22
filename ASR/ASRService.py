import logging
import time

# ASR
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

class ASRService():
    def __init__(self, post_init=True):
        self.models = [
            '../models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
            '../models/speech_paraformer_asr-en-16k-vocab4199-pytorch',
            '../models/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-offline',
        ]

        self.pipes = [None] * len(self.models)

        if not post_init:
            for i in range(len(self.pipes)):
                _ = self.get_asr(i)

    def get_asr(self, actor):
        assert actor >= 0 and actor < len(self.pipes), f"Invalid actor {actor}"

        if not self.pipes[actor]:
            logging.info(f'Initializing ASR Piip {actor}...')
            self.pipes[actor] = pipeline(task=Tasks.auto_speech_recognition, 
                                         model=self.models[actor]) 
            
        return self.pipes[actor]    

    def infer(self, voice_lang, wav_path):
        stime = time.time()
        asr = self.get_asr(voice_lang)
        result = asr(audio_in=wav_path)
        if isinstance(result, dict) and 'text' in result:
            result = result['text']
        print(type(result))
        logging.info('ASR Result: %s. time used %.2f.' % (result, time.time() - stime))
        return str(result)

if __name__ == '__main__':
    service = ASRService()

    # print(wav_path)
    wav_path = 'ASR/test_wavs/0478_00017.wav'
    result = service.infer(wav_path, 0)
    print(result)

    wav_path = 'ASR/test_wavs/asr_example_zh.wav'
    result = service.infer(wav_path, 0)
    print(result)