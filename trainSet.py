#!/usr/bin/env python

import collections
import pandas as pd
import csv
import re

trSet = pd.read_csv('dataset/train-set/trainSet.csv')
DICHOTOMY = ('IE', 'NS', 'TF', 'PJ')
usersByTypes = collections.defaultdict(int)


def writeCSVFile(feature, list):
    with open('dataset/train-set/train' + feature + '.csv', 'w', newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        for fiftyPosts in list:
            row = [post for post in fiftyPosts if
                   ('http' not in post) and (post != '') and (post is not None) and (re.search("[a-zA-Z]", post))]
            if len(row) > 10:
                writer.writerow(row)


for dichotomy in DICHOTOMY:
    d1, d2 = dichotomy
    for mbtiType in trSet['type']:
        if d1 in mbtiType:
            usersByTypes[d1] += 1
        if d2 in mbtiType:
            usersByTypes[d2] += 1

for dichotomy in DICHOTOMY:
    d1, d2 = dichotomy
    limit = min(usersByTypes[d1], usersByTypes[d2])

    listD1 = []
    listD2 = []
    countD1 = 0
    countD2 = 0

    for user in trSet.iterrows():
        if d1 in user[1]['type'] and countD1 < limit:
            listD1.append(user[1]['posts'].split('|||'))
            countD1 += 1

        if d2 in user[1]['type'] and countD2 < limit:
            listD2.append(user[1]['posts'].split('|||'))
            countD2 += 1

        if countD1 >= limit and countD2 >= limit:
            break

    writeCSVFile(d1, listD1)
    writeCSVFile(d2, listD2)
