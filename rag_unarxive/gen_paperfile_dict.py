#!/usr/bin/env python
# coding: utf-8

import tarfile
import pickle
import json
import csv
import os
import re

from tqdm import tqdm
from typing import List

filename = "/data/horse/ws/s9650707-llm_secrets/datasets/unarxive/unarXive_230324_open_subset.tar.xz"
md_target_dir = "/data/horse/ws/s9650707-llm_secrets/datasets/unarxive/md3"

MAX_FILES = None
FILES_IN_TAR = 5599  # only used for progress bar when extracting


def get_jsonl_members(filename: str, max_files: int = None):
    with tarfile.open(filename) as tar_file:
        count = 0
        for tar_member in tar_file:
            yield tar_file.extractfile(tar_member).read()
            count += 1
            if max_files is not None and count >= max_files:
                break

def get_num_files_in_tar(filename: str) -> int:
    with tarfile.open(filename) as tar_file:
        return len(tar_file.getmembers())


def extract_from_json(json_string: str):
    json_obj = json.loads(json_string)
    return json_obj


if FILES_IN_TAR is None:
    FILES_IN_TAR = get_num_files_in_tar(filename)
total_jsonl_files = min(MAX_FILES if MAX_FILES is not None else 1e24, FILES_IN_TAR)

print(f"Total number of jsonl files to be processed: {total_jsonl_files}")


count = 0
title_to_file = {}
for json_string in tqdm(get_jsonl_members(filename, MAX_FILES), total=total_jsonl_files):
    lines = json_string.split(b'\n')

    for line in lines:
        if len(line) == 0:
            continue
        json_obj = extract_from_json(line)

        title = json_obj['metadata']['title']
        title = title.replace('"', '')
        title = title.replace('\n  ', ' ')
        file = f"doc_{count:08d}.md"

        if title in title_to_file:
            #print('WARNING: Title "{}" already exists'.format(title))
            #print('check files {} and {}'.format(title_to_file[title], file))
            title_to_file[title] = title_to_file[title] + [file]
        else:
            title_to_file[title] = [file]

        count += 1

with open('title_to_file.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in title_to_file.items():
       writer.writerow([key, value])
#
#with open('title_to_file.csv') as csv_file:
#    reader = csv.reader(csv_file)
#    mydict = dict(reader)

with open('title_to_file.pkl', 'wb') as f:  # open a text file
    pickle.dump(title_to_file, f)

with open('title_to_file.pkl', 'rb') as f:
    student_file2 = pickle.load(f)


