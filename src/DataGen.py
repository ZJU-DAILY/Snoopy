import time

import fasttext
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import csv
from random import choice

def process_csv_without_nul(file):
    contents = file.read().replace('\x00', '')
    f = StringIO(contents)
    return f


def normalize_rows(matrix):
    norm_matrix = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized_matrix = matrix / norm_matrix
    return normalized_matrix


def text2vec(path_load, path_save, plm='fasttext', normal=0):
    with open(path_load, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data_list = []
        for row in csv_reader:
            data_list.append(row)
    emb_mats = []
    print(len(data_list))

    if plm == 'fasttext':
        ft = fasttext.load_model('../fasttext-english/cc.en.300.bin')
        for id, col in tqdm(enumerate(data_list)): #data_list is a list, each element is a column.
            col_emb = []
            for val in col:
                x = ft.get_sentence_vector(val.replace('\n', ' '))
                # if normal == 1:
                #     x = x / np.linalg.norm(x)
                col_emb.append(x)
            col_emb = np.array(col_emb)
            emb_mats.append(col_emb)
    else:
        model = SentenceTransformer('../huggingface/bert-base-nli-stsb-mean-tokens', device='cuda:0')
        for _, row in tqdm(enumerate(data_list)):
            embeddings = model.encode(row)
            emb_mats.append(embeddings)
    np.save(path_save, emb_mats)


def augment_cell(str, op, ft):
    if op == 'del':
        span_len = random.randint(1, 2)
        l = random.randint(0, len(str)-1)
        if (l+span_len) < len(str):
            new_str = str[:l] + str[l+span_len:]
        else:
            new_str = str
    elif op == 'swap':
        l = random.randint(0, len(str) - 1)
        r = l+random.randint(2, 4)
        if (l+r) < len(str):
            sub_arr = str[l:r + 1]
            str_list = list(sub_arr)
            random.shuffle(str_list)
            sub_arr = ''.join(str_list)
            new_str = str[:l] + sub_arr + str[l + 1:]
        else:
            new_str = str
    elif op == 'insert':
        l = random.randint(0, len(str) - 1)
        insert_list = [",", " ", ".", "_", "#"]
        t = choice(insert_list)
        new_str = str[:l] + t + str[l:]
    elif op == 'repl':
        tokens = str.split()
        if not tokens:
            return str
        random_token = random.choice(tokens)
        new_str = str.replace(random_token, f"{random_token} {random_token}", 1)
    else:
        new_str = str
    x = ft.get_sentence_vector(new_str.replace('\n', ' '))
    y = ft.get_sentence_vector(str.replace('\n', ' '))

    if np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:
        return new_str
    normalized_vector1 = x / np.linalg.norm(x)
    normalized_vector2 = y / np.linalg.norm(y)
    distance = np.linalg.norm(normalized_vector1 - normalized_vector2)
    if distance > 0.2:
        new_str = str
    return new_str


def augment_col(str_list, op_list, ft): # "input a list"
    length = len(str_list)
    x = list(range(0, length-1))
    random.shuffle(x)
    sampled_cell = x[:int(length*random.uniform(0.8, 1))]
    for i in sampled_cell:
        x = random.randint(0, len(op_list)-1)
        if len(str_list[i]) > 0:
            str_list[i] = augment_cell(str_list[i], op_list[x], ft)
    return str_list


def augment_leave(col_list, per, ft):  # "input a col which is a list of str"
    random.shuffle(col_list)
    # anchor_rate = random.uniform(0.5, 1)
    anchor_rate = per
    anchor = col_list[:int(len(col_list) * anchor_rate)]
    leave = col_list[int(len(col_list) * anchor_rate):]
    copy_rate = random.uniform(0.9, 1)
    # if random.random() > 0.3:
    #     copy_rate = random.uniform(0.7, 1)
    # else:
    #     copy_rate = random.uniform(0.4, 7)
    copy = anchor[:int(len(anchor) * copy_rate)]
    if random.random() > 0.5:
        op_list = ["swap", "repl", "insert", "del"]
        copy = augment_col(copy, op_list, ft)
    new_col = copy + leave[int(len(leave) * random.uniform(0.8, 1)):]
    random.shuffle(new_col)
    return anchor, new_col


def augment_text_level(path_load, anchor_path_load, aug_path_load): #text_level使用
    ft = fasttext.load_model('../fasttext-english/cc.en.300.bin')
    with open(path_load, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data_list = []
        for row in csv_reader:
            data_list.append(row)
    anchor_list = list()
    new_col_list = list()
    for i, col in tqdm(enumerate(data_list)):
        anchor_rate = random.uniform(0.6, 1)
        anchor, new_col = augment_leave(col, anchor_rate, ft)
        anchor_list.append(anchor)
        new_col_list.append(new_col)

    with open(aug_path_load, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in new_col_list:
            writer.writerow(row)

    with open(anchor_path_load, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in anchor_list:
            writer.writerow(row)


def augment_mat(mat, tau):
    disturbed_matrix = np.copy(mat)
    for i in range(mat.shape[0]):
        dimension = mat.shape[1]
        noise = 0.02
        if random.random() > 0.5 or np.linalg.norm(mat[i]) == 0:
            disturbed_matrix[i] = mat[i]
        else:
            while True:
                perturbation = np.random.normal(0, noise, dimension)
                original_vec = mat[i]
                disturbed_vec = original_vec + perturbation
                normalized_vector1 = original_vec / np.linalg.norm(original_vec)
                normalized_vector2 = disturbed_vec / np.linalg.norm(disturbed_vec)
                distance = np.linalg.norm(normalized_vector1 - normalized_vector2)
                if distance <= tau:
                    disturbed_matrix[i] = disturbed_vec
                    break
                else:
                    # print("exceed!!! dis={}".format(distance))
                    noise = noise * 0.8
    return disturbed_matrix


def augment_mat_level(path_load, anchor_path, aug_path, y_path, tau=0.1, k=5):
    X = np.load(path_load, allow_pickle=True)
    anchor_list = []
    aug_list = []
    y_list = []
    for i in tqdm(range(len(X))):
        y = []
        mat = X[i]
        np.random.shuffle(mat)
        anchor_rate = random.uniform(0.6, 1)
        anchor = mat[:int(mat.shape[0] * anchor_rate)]
        anchor_list.append(anchor)
        leave = mat[int(mat.shape[0] * anchor_rate):]
        if random.random() > 0.3:
            random_sequence = [random.uniform(0.7, 1) for _ in range(k)]
        else:
            random_sequence = [random.uniform(0.4, 0.7) for _ in range(k)]
        random_sequence.sort(reverse=True)
        for i in range(k):
            copy_rate = random_sequence[i]
            copy = anchor[:int(len(anchor) * copy_rate)]
            # copy = augment_mat(copy, tau)
            new_mat = np.concatenate((copy, leave[:int(len(leave) * random.uniform(0.8, 1))]), axis=0)
            np.random.shuffle(new_mat)
            y.append(str(round(copy_rate, 4)))
            aug_list.append(new_mat)
        y_list.append(y)

    np.save(anchor_path, anchor_list)
    np.save(aug_path, aug_list)

    with open(y_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in y_list:
            writer.writerow(row)

from utils import *

fix_seed(2024)
augment_mat_level("../datasets/Lake/WikiTable/target.npy",
                  "../datasets/Lake/WikiTable/t=0.2/train/mat_level/anchor.npy",
                  "../datasets/Lake/WikiTable/t=0.2/train/mat_level/auglist.npy",
                  "../datasets/Lake/WikiTable/t=0.2/train/mat_level/auglist_y.csv",
                  tau=0.2,  k=3)
# augment_text_level("../datasets/opendata/train/train.csv","../datasets/opendata/train/anchor.csv", "../datasets/opendata/train/auglist.csv")
# text2vec("../datasets/opendata/t=0.2/target.csv", "../datasets/opendata/t=0.2/target.npy")
