#!/bin/bash

rm -fR data
mkdir -p data
curl -X GET --output data/train.csv https://boosters.pro/api/ch/files/pub/HeadHunter_train.csv
curl -X GET --output data/test.csv https://boosters.pro/api/ch/files/pub/HeadHunter_test.csv
curl -X GET --output data/sample_submit.csv https://boosters.pro/api/ch/files/pub/HeadHunter_sample_submit.csv