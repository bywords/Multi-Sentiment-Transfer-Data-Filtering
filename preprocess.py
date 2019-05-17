import argparse
import os, json
from sacremoses import MosesTokenizer
from sklearn.model_selection import train_test_split
from util import transform_score


parser = argparse.ArgumentParser(description='preprocess.py')
parser.add_argument('-data_file', required=True, help='Data file in directory')
parser.add_argument('-data_dir', default="data", help='Directory path to data')


def main(opt):

    dir_name = opt.data_dir
    file_name = opt.data_file

    # MosesTokenizer is particularly used to align the data with the desired input of translation models.
    # See https://github.com/shrimai/Style-Transfer-Through-Back-Translation/issues/18
    mt = MosesTokenizer(lang='en')

    reviews, neutral_reviews = [], []
    scores = []

    with open(os.path.join(dir_name, file_name), 'r', encoding='utf-8') as f:
        _header = f.readline()
        for idx, line in enumerate(f):
            data = line.strip().split('\t')
            score = int(float(data[0]))
            review = data[1]

            if len(review) == 0 or score == 2 or score == 4:
                continue
            if len(review.split()) < 3:
                continue

            tokenized_review_str = mt.tokenize(review, return_str=True)
            if score == 3:
                neutral_reviews.append(tokenized_review_str.lower())
            elif score == 1 or score == 5:
                reviews.append(tokenized_review_str.lower())
                scores.append(score)


    X_train, X_test, Y_train, Y_test = train_test_split(reviews, scores, test_size=0.1, random_state=20180422)
    with open(os.path.join(dir_name, "train.txt"), 'w', encoding='utf-8') as f_classtrain:
        for idx, title in enumerate(X_train):
            score = Y_train[idx]
            new_score = transform_score(score)

            json.dump(dict(score=new_score, headline=title), fp=f_classtrain)
            print(file=f_classtrain)

    with open(os.path.join(dir_name, "test.txt"), 'w', encoding='utf-8') as f_classtest:
        for idx, title in enumerate(X_test):
            score = Y_test[idx]
            new_score = transform_score(score)

            json.dump(dict(score=new_score, headline=title), fp=f_classtest)
            print(file=f_classtest)

    with open(os.path.join(dir_name, "neutral.txt"), 'w', encoding='utf-8') as f_neutral:
        for idx, title in enumerate(neutral_reviews):
            score = 1

            json.dump(dict(score=score, headline=title), fp=f_neutral)
            print(file=f_neutral)

    word_tokens = []
    for title in reviews:
        for token in title.split():
            word_tokens.append(token)
    for title in neutral_reviews:
        for token in title.split():
            word_tokens.append(token)

    from collections import Counter
    counter = Counter(word_tokens)
    with open(os.path.join(dir_name, 'vocab.txt'), 'w', encoding='utf-8') as f:
        for w, cnt in counter.most_common():
            print(w, end=" ", file=f)
            print(cnt, file=f)


if __name__ == '__main__':
    opt = parser.parse_args()
    main(opt)