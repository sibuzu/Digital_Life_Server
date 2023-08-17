import logging

import onnxruntime
from transformers import BertTokenizer
import numpy as np
import onnx

def print_model_graph(model):
    graph = model.graph

    print("Input nodes:")
    for input_node in graph.input:
        print(f"Name: {input_node.name}, Type: {input_node.type}")

    print("\nOutput nodes:")
    for output_node in graph.output:
        print(f"Name: {output_node.name}, Type: {output_node.type}")

    print("\nNodes:")
    for node in graph.node:
        print(f"Name: {node.name}, OpType: {node.op_type}")
        print("Inputs:")
        for input_name in node.input:
            print(f"    {input_name}")
        print("Outputs:")
        for output_name in node.output:
            print(f"    {output_name}")

class SentimentEngine():
    def __init__(self, model_path):
        logging.info('Initializing Sentiment Engine...')
        onnx_model_path = model_path

        self.ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
        #mdl = onnx.load_model(onnx_model_path)
        #print_model_graph(mdl)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def infer(self, text):
        tokens = self.tokenizer(text, return_tensors="np")
        # print("tokens", tokens)
        input_dict = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }
        # Convert input_ids and attention_mask to int64
        input_dict["input_ids"] = input_dict["input_ids"].astype(np.int64)
        input_dict["attention_mask"] = input_dict["attention_mask"].astype(np.int64)
        logits = self.ort_session.run(["logits"], input_dict)[0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        predicted = np.argmax(probabilities, axis=1)[0]
        logging.info(f'Sentiment Engine Infer: {predicted}')
        return predicted

'''
0 開心
1 害怕
2 生氣
3 失落
4 好奇
5 戲謔
'''

if __name__ == '__main__':
    t = ['不许你这样说我，打你',
         '好，我们出发吧',
         '我好无聊',
         '我好高兴',
         '这是什么，好奇怪喔',
         '好像很好玩的样子',
         '这里好黑，好恐怖',
         '我要帮你取一个难听的绰号']
    s = SentimentEngine('models/paimon_sentiment.onnx')
    for x in t:
        r = s.infer(x)
        print(f"{r}: {x}")
