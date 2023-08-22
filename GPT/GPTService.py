import logging
import os
import time

import GPT.machine_id
import GPT.tune as tune


class GPTService():
    def __init__(self, args):
        logging.info('Initializing ChatGPT Service...')

        self.tune = tune.get_tune(args.character, args.model)

        mach_id = GPT.machine_id.get_machine_unique_identifier()
        from revChatGPT.V3 import Chatbot
        if args.APIKey:
            logging.info('you have your own api key. Great.')
            api_key = args.APIKey
        else:
            logging.info('using custom API proxy, with rate limit.')
            os.environ['API_URL'] = "https://api.geekerwan.net/chatgpt2"
            api_key = mach_id

        self.chatbot = Chatbot(api_key=api_key, proxy=args.proxy, system_prompt=self.tune)
        logging.info('API Chatbot initialized.')

    def ask(self, text):
        stime = time.time()
        prev_text = self.chatbot.ask(text)

        logging.info('ChatGPT Response: %s, time used %.2f' % (prev_text, time.time() - stime))
        return prev_text

    def ask_stream(self, text):
        prev_text = ""
        complete_text = ""
        stime = time.time()
        asktext = text

        for data in self.chatbot.ask_stream(text):
            message = data

            if ("。" in message or "！" in message or "？" in message or "\n" in message) and len(complete_text) > 3:
                complete_text += message
                logging.info('ChatGPT Stream Response: %s, @Time %.2f' % (complete_text, time.time() - stime))
                yield complete_text.strip()
                complete_text = ""
            else:
                complete_text += message

            prev_text = data

        if complete_text.strip():
            logging.info('ChatGPT Stream Response: %s, @Time %.2f' % (complete_text, time.time() - stime))
            yield complete_text.strip()
