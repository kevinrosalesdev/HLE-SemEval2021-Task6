import csv
import json
import os

import numpy as np
from tqdm import tqdm

from src.dataprocessing import postprocesser


def get_best_thresholds(dev_data, input_vdata, predictions, tokenizer, filename: str = 'tab-separated-values'):
    best_thresholds = np.zeros(20)
    max_scores = np.zeros(20)
    f1_score = 0
    for threshold in tqdm(np.arange(0, 1.01, 0.01), desc='Getting best thresholds...', ncols=150):
        output = postprocesser.postprocess_data(dev_data, input_vdata, predictions, tokenizer, threshold=threshold)
        with open("out/out-dev.txt", "w") as f:
            json.dump(output, f)

        os.system("python3 SEMEVAL-2021-task6-corpus/scorer/task2/task-2-semeval21_scorer.py "
                  f"-s out/out-dev.txt "
                  f"-r SEMEVAL-2021-task6-corpus/data/dev_set_task2.txt "
                  f"-p SEMEVAL-2021-task6-corpus/techniques_list_task1-2.txt -o > {filename}.csv")

        f1_scores, general_f1_score = read_f1_values(filename)
        if general_f1_score > f1_score:
            f1_score = general_f1_score
            for idx, score in enumerate(f1_scores):
                max_scores[idx] = score
                best_thresholds[idx] = threshold

    return np.array(best_thresholds)


def read_f1_values(filename: str) -> list:
    with open(f"{filename}.csv") as tsv:
        csv_reader = csv.reader(tsv, dialect="excel-tab")
        for row in csv_reader:
            f1_per_class = [float(number) for number in row]
    return f1_per_class[3:], f1_per_class[0]

