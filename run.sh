#!/bin/bash

# 训练模式
python linear_log_model.py \
    --mode train \
    --train_data ag_news_csv/train.csv \
    --stopwords stopwords.txt \
    --epochs 50 \
    --model_path model.pkl \
    --idf_path idf.pkl

# 测试模式
python linear_log_model.py \
    --mode test \
    --test_data ag_news_csv/test.csv \
    --stopwords stopwords.txt \
    --model_path model.pkl \
    --idf_path idf.pkl