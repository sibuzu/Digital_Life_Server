import argparse
import os
import socket
import time
import logging
import traceback
from logging.handlers import TimedRotatingFileHandler
import re
from opencc import OpenCC

import librosa
import requests
import revChatGPT
import soundfile
import sys
import signal
import threading

import GPT.tune
from utils.FlushingFileHandler import FlushingFileHandler
from ASR import ASRService
from GPT import GPTService
from TTS import TTService

import nltk
nltk.download('punkt')  # Download the punkt tokenizer models
from nltk.tokenize import sent_tokenize

console_logger = logging.getLogger()
console_logger.setLevel(logging.INFO)
FORMAT = '%(asctime)s %(levelname)s %(message)s'
console_handler = console_logger.handlers[0]
console_handler.setFormatter(logging.Formatter(FORMAT))
console_logger.setLevel(logging.INFO)
# file_handler = FlushingFileHandler("log.log", encoding='utf-8', formatter=logging.Formatter(FORMAT))
file_handler = logging.FileHandler('log.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter(FORMAT))
file_handler.setLevel(logging.INFO)
console_logger.addHandler(file_handler)
console_logger.addHandler(console_handler)

def ctrlc_handler(signum, frame):
    logging.info("Ctrl+C pressed. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, ctrlc_handler)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def convert_simplified_to_traditional(input_string):
    cc = OpenCC('s2t')  # Specify conversion from Simplified to Traditional
    traditional_string = cc.convert(input_string)
    return traditional_string

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", type=str, nargs='?', required=False)
    parser.add_argument("--password", type=str, nargs='?', required=False)
    parser.add_argument("--accessToken", type=str, nargs='?', required=False)
    parser.add_argument("--proxy", type=str, nargs='?', required=False)
    parser.add_argument("--model", type=str, nargs='?', required=False)
    parser.add_argument("--character", type=str, nargs='?', required=True)
    parser.add_argument("--ip", type=str, nargs='?', required=False)
    parser.add_argument("--port", type=int, default=38434)
    parser.add_argument("--postInit", type=str2bool, nargs='?', required=False)
    return parser.parse_args()


class Server():
    def __init__(self, args):
        # SERVER STUFF
        self.addr = None
        self.conn = None
        logging.info('Initializing Server...')
        self.host = socket.gethostbyname(socket.gethostname())
        self.host = "0.0.0.0"
        self.port = args.port
        logging.info(f"host={self.host}:{self.port}")
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 10240000)
        self.s.bind((self.host, self.port))
        self.tmp_recv_file = 'tmp/server_received.wav'
        self.tmp_proc_file = 'tmp/server_processed.wav'
        self.character = args.character
        self.post_init = args.postInit
     
        logging.info(f"post_init = {self.post_init}")

        # PARAFORMER
        self.asr_service = ASRService.ASRService(self.post_init)

        # CHAT GPT
        self.gpt_service = GPTService.GPTService()

        # TTS
        self.tts_service = TTService.TTService(self.post_init)

    def listen(self):
        # MAIN SERVER LOOP
        while True:
            self.s.listen()
            logging.info(f"Server is listening on {self.host}:{self.port}...")
            self.conn, self.addr = self.s.accept()
            logging.info(f"Connected by {self.addr}")

            char_name = 'character_paimon'
            self.conn.sendall(b'%s' % char_name.encode())
            logging.info('char=%s' % char_name)

            # character = 'character_paimon'
            # self.conn.sendall(b'%s' % character.encode())
            # logging.info('char=%s' % character)
            
            while True:
                try:
                    file, voice_lang, actor = self.__receive_file()
                    assert voice_lang >= 0 and voice_lang <= 2, f'Invalid voice: {voice_lang}'
                    assert actor >= 0 and actor <= 2, f'Invalid actor: {actor}'

                    logging.info(f'file received: {len(file)}')
                    with open(self.tmp_recv_file, 'wb') as f:
                        f.write(file)
                        logging.info('WAV file received and saved.')
                    ask_text = self.process_voice(voice_lang)
                        
                    emo = 0
                    chatbot = self.gpt_service.get_bot(actor)
                    for sentence in chatbot.ask_stream(ask_text):
                        sentence, emo = self.get_emotion(sentence, emo)
                        self.send_voice(actor, sentence, emo, ask_text)
                        ask_text = ''

                    #TEST EMOTION
                    # self.test_emotions()
                    
                    self.notice_stream_end()
                    logging.info('Stream finished.')

                except revChatGPT.typings.APIConnectionError as e:
                    logging.error(e.__str__())
                    logging.info('API rate limit exceeded, sending: %s' % GPT.tune.exceed_reply)
                    self.send_voice(GPT.tune.exceed_reply, 2)
                    self.notice_stream_end()
                except revChatGPT.typings.Error as e:
                    logging.error(e.__str__())
                    logging.info('Something wrong with OPENAI, sending: %s' % GPT.tune.error_reply)
                    self.send_voice(GPT.tune.error_reply, 1)
                    self.notice_stream_end()
                except requests.exceptions.RequestException as e:
                    logging.error(e.__str__())
                    logging.info('Something wrong with internet, sending: %s' % GPT.tune.error_reply)
                    self.send_voice(GPT.tune.error_reply, 1)
                    self.notice_stream_end()
                except Exception as e:
                    logging.error(e.__str__())
                    logging.error(traceback.format_exc())
                    break

    def get_emotion(self, input_string, emo):
        emotions = {
            "exciting": 0,
            "happy": 0,
            "afraid": 1,
            "angry": 2,
            "boring": 3,
            "dispointed": 3,
            "lost": 3,
            "curious": 4,
            "joking": 5
        }

        pattern = r"\[(\w+)\](.*)"
        match = re.match(pattern, input_string)

        if match:
            emotion = match.group(1)
            input_string = match.group(2)
            if emotion in emotions:
                emo = emotions[emotion]
 
        return input_string, emo
    
    def test_emotions(self):
        emo = 0
        test_sentences = [
            '[exciting]這是開心的句子',
            '[afraid]這是害怕的句子',
            '[angry]這是生氣的句子',
            '[boring]這是無聊的句子',
            '[curious]這是好奇的句子',
            '[joking]這是開玩笑的句子',
        ]
        for i in range(60):
            sentence = test_sentences[i%6]
            sentence, emo = self.get_emotion(sentence, emo)
            print("TEST:", sentence, emo)
            self.send_voice(sentence, emo)

    def mode_command(self, input_string):
        pat_D = r"(启动|开始|进入)?(除错|出错)模式"
        pat_C = r"(启动|开始|进入)?(字幕|文字)模式"
        pat_N = r"(启动|开始|进入)?(正常|一般)模式"
        if re.match(pat_D, input_string):
            return "现在进入除错模式"
        elif re.match(pat_C, input_string):
            return "现在进入字幕模式"
        elif re.match(pat_N, input_string):
            return "现在进入正常模式"
        else:
            return ""

    def notice_stream_end(self):
        time.sleep(0.5)
        self.conn.sendall(b'stream_finished')

    def send_voice(self, actor, resp_text, senti = 0, ask_text=None):
        tts = self.tts_service.get_tts(actor)
        tts.read_save(resp_text, self.tmp_proc_file)

        with open(self.tmp_proc_file, 'rb') as f:
            senddata = f.read()

        if actor == 0:
            resp_text = convert_simplified_to_traditional(resp_text)
            if ask_text:
                ask_text = convert_simplified_to_traditional(ask_text)
        
        if actor==1: # english
            resp_text = self.split_sentences(resp_text.strip())
        
        if ask_text:
            resp_text = ask_text + '<sp>' + resp_text  
        str_bdata = b'%s' % resp_text.encode('utf-16-le')
        n = len(str_bdata)
        senddata += str_bdata
        senddata += b'%c%c' % (n//256, n%256)
        senddata += b'?!'
        senddata += b'%c' % senti
        self.conn.sendall(senddata)
        time.sleep(0.5)
        logging.info(f'WAV SENT, {resp_text}, {senti}, size =  {len(resp_text)}, {len(str_bdata)}, {len(senddata)}')

    def split_sentences(self, paragraph):
        sents = sent_tokenize(paragraph)
        return '\n'.join(sents[::-1])

    def __receive_file(self):
        file_data = b''
        voice = 0
        actor = 0
        while True:
            data = self.conn.recv(1024)
            self.conn.send(b'sb')
            if data[-3:-1] == b'?!':
                file_data += data[0:-3]
                voice = data[-1] // 10
                actor = data[-1] % 10
                break
            if not data:
                # logging.info('Waiting for WAV...')
                time.sleep(0.1)
                continue
            file_data += data

        logging.info(f"receive voice={voice}, actor={actor}, len={len(file_data)}")
        return file_data, voice, actor

    def fill_size_wav(self):
        with open(self.tmp_recv_file, "r+b") as f:
            # Get the size of the file
            size = os.path.getsize(self.tmp_recv_file) - 8
            # Write the size of the file to the first 4 bytes
            f.seek(4)
            f.write(size.to_bytes(4, byteorder='little'))
            f.seek(40)
            f.write((size - 28).to_bytes(4, byteorder='little'))
            f.flush()

    def process_voice(self, voice_lang):
        # stereo to mono
        self.fill_size_wav()
        y, sr = librosa.load(self.tmp_recv_file, sr=None, mono=False)
        y_mono = librosa.to_mono(y)
        y_mono = librosa.resample(y_mono, orig_sr=sr, target_sr=16000)
        soundfile.write(self.tmp_recv_file, y_mono, 16000)
        text = self.asr_service.infer(voice_lang, self.tmp_recv_file)

        return text

def server_task():
    try:
        args = parse_args()
        s = Server(args)
        s.listen()
    except Exception as e:
        logging.error(e.__str__())
        logging.error(traceback.format_exc())
        raise e 

if __name__ == '__main__':
    server_thread = threading.Thread(target=server_task)
    server_thread.daemon = True
    server_thread.start()

    while True:
        time.sleep(1)