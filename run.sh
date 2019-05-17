#!/usr/bin/env bash
python preprocess.py --data_file yelp_restaurants_review_star_sentence_fixed.tsv
python train_binary_cnn.py
