#!/usr/bin/env python

import collections
import pandas as pd
import csv
import re

tsSet = pd.read_csv('dataset/test-set/testSet.csv')
FEATURES = ('I', 'E', 'N', 'S', 'T', 'F', 'P', 'J')
usersByTypes = collections.defaultdict(list)

for user in tsSet.iterrows():
    usersByTypes[user[1]['type']].append(user[1]['posts'].split('|||'))

for feature in FEATURES:
    with open('dataset/test-set/test' + feature + '.csv', 'w', newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        for mbtiType in usersByTypes.keys():
            if feature in mbtiType:
                for fiftyPosts in usersByTypes[mbtiType]:
                    row = [post for post in fiftyPosts if
                           ('http' not in post) and (post != '') and (post is not None) and (
                               re.search("[a-zA-Z]", post))]
                    if len(row) > 10:
                        writer.writerow(row)
