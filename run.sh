#!/usr/bin/env bash
python preprocess.py -data_file amazon_finefood_reviews_score_sentence.tsv
python train_binary_cnn.py --max_epochs 5
python prepare_final_dataset.py -output_file amazon_final_data_unfiltered.tsv -filter 0
python prepare_final_dataset.py -output_file amazon_final_data_filtered.tsv -filter 1
