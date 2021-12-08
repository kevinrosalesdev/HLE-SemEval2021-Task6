from transformers import RobertaTokenizer
from nltk.tokenize import TreebankWordTokenizer
import numpy as np


def is_contiguous(a: int, b: int) -> bool:
    return a == b+1 or b == a+1


def extract_sequences(indexes: list) -> list:
    res = []
    temp = [indexes[0]]
    for i in indexes[1:]:
        if is_contiguous(i, temp[-1]):
            temp.append(i)
        else:
            res.append(temp)
            temp = [i]
    res.append(temp)
    return res


def adjust_output(tokenizer: RobertaTokenizer, sentence: str, output: np.ndarray) -> np.array:
    res = []
    tokenizer_match = tokenizer.tokenize(sentence)
    temp = [output[0]]
    for idx, token in enumerate(tokenizer_match[1:], 1):
        if token[0] == 'Ġ':
            res.append(np.mean(temp, axis=0))
            temp = [output[idx]]
        else:
            temp.append(output[idx])

    res.append(np.mean(temp, axis=0))

    return np.array(res)


def postprocess_data(metadata: list, input_data: list, model_prediction: list,
                     tokenizer: RobertaTokenizer, threshold: float) -> list:
    output = []
    labels_file = "SEMEVAL-2021-task6-corpus/techniques_list_task1-2.txt"
    label_dict = {}
    with open(labels_file, "r") as f:
        for idx, label in enumerate(f.readlines()):
            label_dict[idx] = label.strip()

    twt = TreebankWordTokenizer()
    for idx, meme in enumerate(metadata):
        output_i = {"id": meme["id"],
                    "text": meme["text"]}

        spans = list(twt.span_tokenize(' '.join(input_data[idx])))
        prediction_i = adjust_output(tokenizer, ' '.join(input_data[idx]), model_prediction[idx])
        labels = []

        for propaganda_technique in range(20):
            accepted_words = np.where(prediction_i[:, propaganda_technique] > threshold)[0]

            if accepted_words.size > 0:
                contiguous_techniques = extract_sequences(accepted_words)
            else:
                continue

            for sequence in contiguous_techniques:
                labels.append({"technique": label_dict[propaganda_technique],
                               "start": spans[sequence[0]][0],
                               "end": spans[sequence[-1]][1]})

        output_i["labels"] = labels
        output.append(output_i)

    return output