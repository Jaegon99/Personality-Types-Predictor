#!/usr/bin/env python

import collections
import pandas as pd
import csv

# Myers Briggs personality dataset
ds = pd.read_csv('dataset/mbti_1.csv')

# Myers Briggs percentages for different personality types
percentages = {'ISTJ': 0.12, 'ISFJ': 0.14, 'INFJ': 0.02, 'INTJ': 0.02,
               'ISTP': 0.05, 'ISFP': 0.09, 'INFP': 0.04, 'INTP': 0.03,
               'ESTP': 0.04, 'ESFP': 0.08, 'ENFP': 0.08, 'ENTP': 0.03,
               'ESTJ': 0.09, 'ESFJ': 0.12, 'ENFJ': 0.03, 'ENTJ': 0.02}

usersByTypes = collections.defaultdict(int)
for mbtiType in ds['type']:
    usersByTypes[mbtiType] += 1

limitingType = None
minSize = float('infinity')

for mbtiType in usersByTypes.keys():
    size = usersByTypes[mbtiType] / percentages[mbtiType]
    if size < minSize:
        minSize = size
        limitingType = mbtiType

dataset = collections.defaultdict(list)
for row in ds.iterrows():
    dataset[row[1]['type']].append(row)

uncleanList = []

with open('dataset/test_set/testSet.csv', 'w', newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(['type', 'posts'])

    for mbti in percentages.keys():
        typeList = dataset[mbti]
        for i in range(0, int(round(minSize * percentages[mbti]))):
            writer.writerow(typeList[i][1])
        uncleanList.append(typeList[int(round(minSize * percentages[mbti])): len(typeList)])

with open('dataset//train_set/trainSet.csv', 'w', newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(['type', 'posts'])

    for mbti in uncleanList:
        for i in mbti:
            writer.writerow(i[1])
