#!/usr/bin/env bash
#python preprocess.py -data_file yelp_restaurants_review_star_sentence_fixed.tsv
python train_binary_cnn.py --max_epochs 5
#python prepare_final_dataset.py -output_file yelp_final_data.tsv
