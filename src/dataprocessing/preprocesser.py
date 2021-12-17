import os.path

from nltk.tokenize import TreebankWordTokenizer, word_tokenize
from os import walk

import json
import numpy as np
import csv

TECHNIQUE_CONV_DICT = {'Appeal_to_Authority': 'Appeal to authority',
                       'Appeal_to_fear-prejudice': 'Appeal to fear/prejudice',
                       'Bandwagon': 'Bandwagon',
                       'Black-and-White_Fallacy': 'Black-and-white Fallacy/Dictatorship',
                       'Causal_Oversimplification': 'Causal Oversimplification',
                       'Doubt': 'Doubt',
                       'Exaggeration,Minimisation': 'Exaggeration/Minimisation',
                       'Flag-Waving': 'Flag-waving',
                       'Loaded_Language': 'Loaded Language',
                       'Name_Calling,Labeling': 'Name calling/Labeling',
                       'Obfuscation,Intentional_Vagueness,Confusion': 'Obfuscation, Intentional vagueness, Confusion',
                       'Red_Herring': 'Presenting Irrelevant Data (Red Herring)',
                       'Reductio_ad_hitlerum': 'Reductio ad hitlerum',
                       'Repetition': 'Repetition',
                       'Slogans': 'Slogans',
                       'Straw_Men': 'Smears',
                       'Thought-terminating_Cliches': 'Thought-terminating cliché',
                       'Whataboutism': 'Whataboutism'}


def convert_name_technique(name_tech: str) -> str:
    return TECHNIQUE_CONV_DICT[name_tech]


def is_span1_within_span2(span1: tuple, span2: tuple) -> bool:
    return span1[0] >= span2[0] and span1[1] <= span2[1]


def transform_2020_data(train_filepath: str, labels_filepath: str):
    try:
        f_train = open(f'{train_filepath}.txt', "r")
        f_labels = open(f'{labels_filepath}.labels', "r")
    except:
        raise Exception('At least one of the filepaths is not correct')

    sentences_news = f_train.readlines()
    sentences_news_cleaned = [line.rstrip() for line in sentences_news]
    spans_sentences = []
    end_span = 0
    for sentence in sentences_news_cleaned:
        if end_span == 0:
            start_span = 0
        else:
            start_span = end_span + 1  # 1 for avoid overlapping, the other for the \n in between sentences.

        end_span = start_span + len(sentence)
        spans_sentences.append((start_span, end_span))

    labels = [[line[1], int(line[2]), int(line[3])] for line in csv.reader(f_labels, dialect="excel-tab")]
    converted_news = []

    for idx, sentence in enumerate(sentences_news_cleaned):
        converted_new = {'id': f'{os.path.basename(train_filepath)}-{idx}', 'text': sentence}
        converted_labels = []
        for label in labels:
            spans_label = label[1], label[2]
            spans_sentence = spans_sentences[idx]
            if is_span1_within_span2(spans_label, spans_sentence):
                if label[0] == 'Whataboutism,Straw_Men,Red_Herring':
                    techniques = [convert_name_technique(label[0].split(',')[0]),
                                  convert_name_technique(label[0].split(',')[1]),
                                  convert_name_technique(label[0].split(',')[2])]

                elif label[0] == 'Bandwagon,Reductio_ad_hitlerum':
                    techniques = [convert_name_technique(label[0].split(',')[0]),
                                  convert_name_technique(label[0].split(',')[1])]

                else:
                    techniques = [convert_name_technique(label[0])]

                for technique in techniques:
                    converted_label = {'start': spans_label[0] - spans_sentence[0],
                                       'end': spans_label[1] - spans_sentence[0],
                                       'technique': technique,
                                       'text_fragment': sentence[spans_label[0] - spans_sentence[0]:
                                                                 spans_label[1] - spans_sentence[0]]}

                    converted_labels.append(converted_label)

        if len(converted_labels) > 0:
            converted_new['labels'] = converted_labels
            converted_news.append(converted_new)

    f_train.close()
    f_labels.close()
    return converted_news


def augment_data(article_dir: str, label_dir: str, output_filepath:str):
    articles = next(walk(article_dir), (None, None, []))[2]  # [] if no file
    labels = next(walk(label_dir), (None, None, []))[2]  # [] if no file
    if not articles or not labels:
        raise Exception('At least one of the provided paths is not correct.')

    new_corpus = list()
    for article in articles:
        article_name = article.split('.')[0]
        label_name = article_name + ".task2-TC"
        article_path = article_dir + f'/{article_name}'
        label_path = label_dir + f'/{label_name}'
        new_corpus.extend(transform_2020_data(article_path, label_path))

    with open(f'{output_filepath}.json', 'w', encoding='utf8') as f:
        json.dump(new_corpus, f, ensure_ascii=False)
        f.close()


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

