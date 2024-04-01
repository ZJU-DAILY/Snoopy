import math
from io import StringIO

import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.io import loadmat
import random
import os
import torch
import logging
import torch.nn.functional as F


logger = logging.getLogger(__name__)
def fix_seed(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def save_model(model, folder_name, version): #todo
    torch.save(model.state_dict(), (folder_name + version + "trained.pth"))
    print("Model checkpoint saved!")


def cal_score(query, target, t=0.2):
    query_vectors = torch.tensor(query).to('cuda:0')
    target_vectors = torch.tensor(target).to('cuda:0')
    query_vectors = F.normalize(query_vectors, p=2, dim=1)
    target_vectors = F.normalize(target_vectors, p=2, dim=1)
    distance_squared = torch.sum((query_vectors[:, None] - target_vectors) ** 2, dim=2)
    euclidean_distances = torch.sqrt(distance_squared)
    values = torch.min(euclidean_distances, dim=1)[0]
    # values, topk_indices = torch.topk(euclidean_distances, k=1, dim=1, largest=False)
    binary_values = torch.where(values <= t, torch.tensor(1).to('cuda:0'), torch.tensor(0).to('cuda:0'))
    result = torch.sum(binary_values)
    return (result*1.0/query.shape[0]).item()


def cal_score_cos(query, target):
    query_vectors = torch.tensor(query)
    target_vectors = torch.tensor(target)
    similarity_matrix = F.cosine_similarity(query_vectors[:, None, :], target_vectors[None, :, :], dim=2)
    values, topk_indices = torch.topk(similarity_matrix, k=1, dim=1)
    binary_values = torch.where(values > 0.9, torch.tensor(1), torch.tensor(0))
    result = torch.sum(binary_values)
    return (result*1.0/query.shape[0]).item()


def cal_NDCG(query, lake, pred, gt, k):
    ndcg = 0
    for i in range(len(query)):
        dcg_pred = DCG(query[i], lake[pred[i].tolist()], k)
        dcg_ideal = DCG(query[i], lake[[int(x) for x in gt[i]]], k)
        ndcg += dcg_pred / dcg_ideal
    return ndcg/len(query)


def DCG(query, target_set, k):
    dcg = 0
    for i in range(min(k, len(target_set))):
        dcg += cal_score(query, target_set[i])/math.log2(i + 2)
    return dcg



def load_query(path):
    query = np.load(path, allow_pickle=True)

    return query


def process_csv_without_nul(file):
    contents = file.read().replace('\x00', '')
    f = StringIO(contents)
    return f

def parse_options(parser):
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--plm', type=str, default='fasttext')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--queue_length', type=int, default=32)
    parser.add_argument('--n_proxy_sets', type=int, default=90)
    parser.add_argument('--n_elements', type=int, default=50)
    parser.add_argument('--d', type=int, default=300, help='dimension of each cell')
    parser.add_argument('--t', type=float, default=0.08)
    parser.add_argument('--momentum', type=float, default=0.9999)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--rank', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--topk', type=int, default=25)
    parser.add_argument('--datasets', type=str, default='WikiTable')
    parser.add_argument('--test_path_mat', type=str, default="../datasets/Lake/WikiTable/target.npy")
    parser.add_argument('--query', type=str, default="../datasets/Lake/WikiTable/25_30_query_0.2_final.npy")
    parser.add_argument('--gt', type=str, default="../datasets/Lake/WikiTable/t=0.2/test/index.csv")
    parser.add_argument('--model_mat', type=str, default="../check/" + "WikiTable/" + "t=0.2_gendata_mat_leveltrained.pth")
    parser.add_argument('--anchor_path', type=str, default="../datasets/Lake/WikiTable/t=0.2/train/gen/mat_level/anchor.npy")
    parser.add_argument('--auglist_path', type=str, default="../datasets/Lake/WikiTable/t=0.2/train/gen/mat_level/auglist.npy")
    parser.add_argument('--list_size', type=int, default=3)
    parser.add_argument('--da', type=str, default='True')

    return parser.parse_args()