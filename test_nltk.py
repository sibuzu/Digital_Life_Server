import nltk
nltk.download('punkt')  # Download the punkt tokenizer models

from nltk.tokenize import sent_tokenize

def split_sentences(sentences):
    results = []
    for x in list(sentences):
        if len(x) > 30 and ',' in x:
            results += x.split(',')
        else:
            results.append(x)
    return results

paragraph = "Remember to place 4.5 this code before any code that uses the imported module, as the logging level is set once and affects subsequent log messages. Also, be aware that this only affects the logging behavior of the specified module. Is it right? Terrible!!! If other modules or parts of your code are producing log messages, they won't be affected by this change unless you modify their logger levels as well."

sentences = sent_tokenize(paragraph)
sentences = split_sentences(sentences)

for sentence in sentences:
    print(sentence)
