import sys
import time

sys.path.append('TTS/vits')

import soundfile
import os
os.environ["PYTORCH_JIT"] = "0"
import torch

import TTS.vits.commons as commons
import TTS.vits.utils as utils

from TTS.vits.models import SynthesizerTrn
from TTS.vits.text.symbols import symbols
from TTS.vits.text import text_to_sequence, create_symbol_id_map

import logging
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


class TTSEngin():
    def __init__(self):
        self.engine = None
        self.speed = 1
        self.name = 'tts'
        self.hps = None
        self.symbol_to_id = None
        self.id_to_symbol = None
        self.tag = ''

    def get_text(self, text):
        text_norm = text_to_sequence(text, self.hps.data.text_cleaners, self.symbol_to_id)
        if self.hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def read(self, text):
        text = text.replace('~', '！')
        text = f'{self.tag}{text}{self.tag}'
        stn_tst = self.get_text(text)

        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            audio = self.engine.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.2, length_scale=self.speed)[0][
                0, 0].data.cpu().float().numpy()
        return audio

    def read_save(self, text, filename):
        stime = time.time()
        au = self.read(text)
        soundfile.write(filename, au, self.hps.data.sampling_rate)
        logging.info('VITS Synth Done, time used %.2f' % (time.time() - stime))


class TTService():
    def __init__(self, post_init=True):
        ## hard coded character map
        self.cfgs = [
            ['TTS/models/paimon6k.json', 'TTS/models/paimon6k_390k.pth', 'paimon', 1.2, ''],
            ['TTS/models/ljs_base.json', 'TTS/models/pretrained_ljs.pth', 'ljs_base', 1, ''],
            ['TTS/models/momoi.json', 'TTS/models/momoi.pth', 'momoi', 1, '[JA]']
        ]

        self.tts = [None] * len(self.cfgs)
        if not post_init:
            for i in range(len(self.tts)):
                _ = self.get_tts(i)

    def get_tts(self, actor):
        assert actor>=0 and actor<len(self.tts), f"Invalid actor: {actor}"

        if not self.tts[actor]:
            logging.info(f'Create TTS of Actor{actor}...')

            tts = TTSEngin()
            self.tts[actor] = tts

            cfg, model, tts.name, tts.speed, tts.tag = self.cfgs[actor]
            tts.hps = utils.get_hparams_from_file(cfg)
            
            tts.symbol_to_id, tts.id_to_symbol = create_symbol_id_map(tts.hps.symbols)
        
            tts.engine = SynthesizerTrn(
                len(tts.hps.symbols),
                tts.hps.data.filter_length // 2 + 1,
                tts.hps.train.segment_size // tts.hps.data.hop_length,
                **tts.hps.model).cuda()
            _ = tts.engine.eval()
            _ = utils.load_checkpoint(model, tts.engine, None)
        
        return self.tts[actor]
    



