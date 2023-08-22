import logging
import time

import GPT.tune as tune
from revChatGPT.V3 import Chatbot

class GPTBot():
    def __init__(self):
        self.engine = None
        self.count = 0
        self.tune = ''

    def ask(self, text):
        
        stime = time.time()
        prev_text = self.engine.ask(text)

        logging.info('ChatGPT Response: %s, time used %.2f' % (prev_text, time.time() - stime))
        return prev_text

    def ask_stream(self, text):
        complete_text = ""
        stime = time.time()

        self.count = (self.count + 1) % 5
        if self.count == 0:
            text = self.tune + '\n' + text

        for data in self.engine.ask_stream(text):
            message = data

            if ("。" in message or "！" in message or "？" in message or "\n" in message) and len(complete_text) > 3:
                complete_text += message
                logging.info('ChatGPT Stream Response: %s, @Time %.2f' % (complete_text, time.time() - stime))
                yield complete_text.strip()
                complete_text = ""
            else:
                complete_text += message

        if complete_text.strip():
            logging.info('ChatGPT Stream Response: %s, @Time %.2f' % (complete_text, time.time() - stime))
            yield complete_text.strip()

    def retune(self):
        logging.info(f'Retune: {self.tune}')
        self.engine.ask(self.tune, role="system")

class GPTService():
    def __init__(self):
        self.bots = [None, None, None]
        self.configs = [
            ('GPT/prompts/paimon35-cn.txt',
            'GPT/prompts/paimon35-cn2.txt',
            'sk-6nV0u0HGb5x2DGiA9LajT'
            '3BlbkFJf2hr1jr2yDnL8B3eqEb6'),
            ('GPT/prompts/paimon35-en.txt',
            'GPT/prompts/paimon35-en2.txt',
            'sk-XY3aVRkPNZESDKUkkTjoT'
            '3BlbkFJIjRTmbV4FaUW2k4CFixT'),
            ('GPT/prompts/paimon35-ja.txt',
            'GPT/prompts/paimon35-ja2.txt',
            'sk-uSsBtY2xCwBuXIJFdyfrT'
            '3BlbkFJKihFqvsnB8DJyEx2XwtC'),
        ]

    def get_bot(self, actor):
        assert actor>=0 and actor<len(self.bots), f"Invalid actor: {actor}"

        if not self.bots[actor]:
            bot = GPTBot()
            self.bots[actor] = bot

            logging.info(f'Create Chatbot of Actor{actor}...')
            gtune = open(self.configs[actor][0], 'r', encoding='utf-8').read() 
            bot.tune =   open(self.configs[actor][1], 'r', encoding='utf-8').read()  
            api_key =  self.configs[actor][2]       
            logging.info(f"tune{actor}={gtune}")
            bot.engine = Chatbot(api_key=api_key, system_prompt=gtune)
            logging.info(f'API Chatbot Actor{actor} initialized.')
        
        return self.bots[actor]

