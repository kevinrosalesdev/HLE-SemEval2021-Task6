from nltk.tokenize import TreebankWordTokenizer, word_tokenize

import numpy as np


def preprocess_data(input_data: list):
    """
    Preprocess each sentence so as to get the input tokenized per word and
    the output as a multihot encoding.

    :param input_data: Array of dicts from the corpus.
    :return: input_text, output_multihot
    """
    input_text = []
    output_multihot = []
    tags = 20
    twt = TreebankWordTokenizer()
    labels_file = "SEMEVAL-2021-task6-corpus/techniques_list_task1-2.txt"
    label_dict = {}
    with open(labels_file, "r") as f:
        for idx, label in enumerate(f.readlines()):
            label_dict[label.strip()] = idx

    for meme in input_data:
        spans_meme = list(twt.span_tokenize(meme["text"].lower()))
        text_meme = word_tokenize(meme["text"].lower())
        len_meme = len(text_meme)
        input_text.append(text_meme)
        output_meme = np.zeros((len_meme, tags))
        for label in meme['labels']:
            target_span = (label['start'], label['end'])
            for idx, span_meme in enumerate(spans_meme):
                if span_meme[0] >= target_span[0] and span_meme[1] <= target_span[1]:
                    output_meme[idx][label_dict[label["technique"]]] = 1
                if span_meme[1] > target_span[1]:
                    break

        output_multihot.append(output_meme)
    return input_text, output_multihot

