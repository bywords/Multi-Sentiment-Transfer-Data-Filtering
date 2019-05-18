import os, json
import random
import argparse
from sklearn.model_selection import train_test_split

random.seed(20180422)

parser = argparse.ArgumentParser(description='prepare_final_dataset.py')
parser.add_argument('-data_dir', default="data", help='Directory path to data')
parser.add_argument('-output_dir', default="output", help='Final output path')
parser.add_argument('-data_file', default="train.txt", help='Data file of negative and positive sentiments')
parser.add_argument('-neutral_file', default="neutral_filtered.txt", help='Neutral review with prediction scores')
parser.add_argument('-output_file', required=True, help='Final output data')


def filter_neutral_by_confidence(path, N):
    neutral_data = []
    with open(path, 'r', encoding='utf-8') as f:
        for string_ in f:
            dict_example = json.loads(string_)
            headline = dict_example["headline"]
            output = float(dict_example["output"])

            conf_value = abs(output-0.5)  # value between 0 - 0.5

            neutral_data.append((headline, conf_value))

    neutral_list_dic = []
    for headline, conf_value in sorted(neutral_data, key=lambda x: x[1])[:N]:
        neutral_list_dic.append(dict(score=1, headline=headline))

    return neutral_list_dic


def load_positive_negative(path, N):
    positive_list_dic = []
    negative_list_dic = []
    with open(path, 'r', encoding='utf-8') as f:
        for string_ in f:
            dict_example = json.loads(string_)

            if dict_example['score'] == 2:
                positive_list_dic.append(dict_example)
            elif dict_example['score'] == 0:
                negative_list_dic.append(dict_example)

    random.shuffle(positive_list_dic)
    random.shuffle(negative_list_dic)

    return positive_list_dic[:N], negative_list_dic[:N]


def store_output(data_list, output_path):

    Xs, ys = [], []
    for dic in data_list:
        y = dic['score']
        X = dic['headline']

        Xs.append(X)
        ys.append(y)

    with open(output_path, 'w', encoding='utf-8') as f:
        print("score", end='\t', file=f)
        print("text", file=f)

        for dic in data_list:
            y = dic['score']
            X = dic['headline']

            print(X, end='\t', file=f)
            print(y, file=f)


def main(opt):

    neutral_path = os.path.join(opt.data_dir, opt.neutral_file)
    neutral_list = filter_neutral_by_confidence(neutral_path, N=80000)
    positive_negative_path = os.path.join(opt.data_dir, opt.data_file)
    positive_list, negative_list = load_positive_negative(positive_negative_path, N=80000)

    if not os.path.isdir(opt.output_dir):
        os.mkdir(opt.output_dir)
    output_path = os.path.join(opt.output_dir, opt.output_file)
    store_output(positive_list+neutral_list+negative_list, output_path)


if __name__ == '__main__':
    opt = parser.parse_args()
    main(opt)
