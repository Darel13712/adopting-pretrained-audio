import sys
import os
import torch
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import LabelEncoder
from utils import *


def calc_knn(model_name, dataset, k=100, suffix="cosine"):
    run_name = f'{dataset}_{model_name}_{suffix}'
    print(run_name)
    writer = SummaryWriter(log_dir=f'runs/{run_name}')

    train, val, hot_test, cold_test, ue, ie = load_dataset(dataset)
    mixed_test = pd.concat([hot_test, cold_test], ignore_index=True)
    user_history = train.groupby('user_id')['item_id'].agg(set).to_dict()

    item_embs = np.load(f'/gpfs/space/projects/music_ca/DATA/{"music4all" if dataset == "m4a" else dataset}/embeddings/{model_name}.npy')
    user_embs = np.stack(train.groupby('user_id')['item_id'].apply(lambda items: item_embs[items].mean(axis=0)).values)

    unique_track_id = np.sort(np.unique(np.concatenate((train.track_id.unique(), cold_test.track_id.unique()))))
    item_embs = item_embs[unique_track_id] # this essentially converts initial track_ids to item_ids for embs

    if suffix == 'cosine':
        user_embs = user_embs / np.linalg.norm(user_embs, axis=1, keepdims=True)
        item_embs = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(item_embs.shape[1])
    index.add(item_embs)

    index_cold = faiss.IndexIDMap(faiss.IndexFlatIP(item_embs.shape[1]))
    index_cold.add_with_ids(item_embs[cold_test.item_id.unique()], cold_test.item_id.unique())

    # all_test_users =  np.unique(np.concatenate((val.user_id.unique(), hot_test.user_id.unique(), cold_test.user_id.unique())))
    all_test_users =  np.unique(np.concatenate((hot_test.user_id.unique(), cold_test.user_id.unique())))
    mixed_recommendations = {}
    hot_recommendations = {}
    cold_recommendations = {}
    cold_items = set(cold_test.item_id.unique())

    for user_id in tqdm(all_test_users):
        history = user_history[user_id]
        user_vector = user_embs[user_id]
        distances, indices = index.search(np.array([user_vector]), k + len(history) + len(cold_items))
        base_recs = [idx for idx in indices[0] if idx not in history]
        hot_recommendations[user_id] = [item_id for item_id in base_recs if item_id not in cold_items][:k]

        distances, indices = index_cold.search(np.array([user_vector]), k + len(history))
        cold_recommendations[user_id] = [idx for idx in indices[0] if idx not in history][:k]


    hot_pred = dict_to_pandas(hot_recommendations)
    hot_pred.to_parquet(f'preds/{run_name}_hot.pqt')

    cold_pred = dict_to_pandas(cold_recommendations)
    cold_pred.to_parquet(f'preds/{run_name}_cold.pqt')

    # metrics_val = calc_metrics(val, hot_pred, False)
    # metrics_val.to_parquet(f'metrics/{run_name}_val.pqt')
    # metrics_val = metrics_val.apply(mean_confidence_interval)
    # metrics_val.index = ['mean', 'conf']
    # metrics_val.to_csv(f'metrics/{run_name}_val.csv')
    # print('Val metrics:')
    # print(metrics_val)
    # for metric_name, metric_value in metrics_val.items():
    #     writer.add_scalar(f'Val/{metric_name}', metric_value['mean'], 0)

    metrics_test = calc_metrics(hot_test, hot_pred, False)
    metrics_test.to_parquet(f'metrics/{run_name}_hot_test.pqt')
    metrics_test = metrics_test.apply(mean_confidence_interval)
    metrics_test.index = ['mean', 'conf']
    metrics_test.to_csv(f'metrics/{run_name}_hot_test.csv')
    print('Hot test metrics:')
    print(metrics_test)

    for metric_name, metric_value in metrics_test.items():
        writer.add_scalar(f'Hot test/{metric_name}', metric_value['mean'], 0)

    metrics_test = calc_metrics(cold_test, cold_pred, False, 20)
    metrics_test.to_parquet(f'metrics/{run_name}_cold_test.pqt')
    metrics_test = metrics_test.apply(mean_confidence_interval)
    metrics_test.index = ['mean', 'conf']
    metrics_test.to_csv(f'metrics/{run_name}_cold_test.csv')
    print('Cold test metrics:')
    print(metrics_test)

    for metric_name, metric_value in metrics_test.items():
        writer.add_scalar(f'Cold test/{metric_name}', metric_value['mean'], 0)

    writer.close()

os.makedirs('runs', exist_ok=True)
os.makedirs('metrics', exist_ok=True)


for dataset in ['m4a']:
    print(dataset)
    for model in tqdm(['musicnn', 'encodecmae', 'jukemir', 'music2vec', 'mert', 'musicfm', 'mfcc']):
        calc_knn(model, dataset)
