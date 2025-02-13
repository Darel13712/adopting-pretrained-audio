import pandas as pd
import numpy as np
import scipy
from rs_metrics import hitrate, mrr, precision, recall, ndcg
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import LabelEncoder


def load_dataset(dataset):
    project_folder = '~/pretrained_tors/'
    train = pd.read_parquet(project_folder + f'data/{dataset}/train.pqt')
    val = pd.read_parquet(project_folder + f'data/{dataset}/val.pqt')
    hot_test = pd.read_parquet(project_folder + f'data/{dataset}/hot_test.pqt')
    cold_test = pd.read_parquet(project_folder + f'data/{dataset}/cold_test.pqt')


    train['item_id'] = train['track_id']
    val['item_id'] = val['track_id']
    hot_test['item_id'] = hot_test['track_id']
    cold_test['item_id'] = cold_test['track_id']

    train, val, hot_test, cold_test, ue, ie = encode_ids(train, val, hot_test, cold_test)
    return train, val, hot_test, cold_test, ue, ie

def encode_ids(train, val, hot_test, cold_test):
    all_users = train.user_id.unique()
    all_items = np.concatenate([
        train.item_id.unique(), 
        cold_test.item_id.unique()
        ])

    ue = LabelEncoder()
    ie = LabelEncoder()

    ue.fit(all_users)
    ie.fit(all_items)

    train['user_id'] = ue.transform(train['user_id'])
    train['item_id'] = ie.transform(train['item_id'])
    val['user_id'] = ue.transform(val['user_id'])
    val['item_id'] = ie.transform(val['item_id'])
    hot_test['user_id'] = ue.transform(hot_test['user_id'])
    hot_test['item_id'] = ie.transform(hot_test['item_id'])
    cold_test['user_id'] = ue.transform(cold_test['user_id'])
    cold_test['item_id'] = ie.transform(cold_test['item_id'])

    return train, val, hot_test, cold_test, ue, ie

def dict_to_pandas(d, key_col='user_id', val_col='item_id'):
    return (
        pd.DataFrame(d.items(), columns=[key_col, val_col])
            .explode(val_col)
            .reset_index(drop=True)
    )

def calc_metrics(test, pred, mean=True, k=50):
    metrics = pd.DataFrame()
    metrics[f'HitRate@{k}'] = hitrate(test, pred, k=50, apply_mean=mean)
    metrics[f'Recall@{k}'] = recall(test, pred, k=50, apply_mean=mean)
    metrics[f'NDCG@{k}'] = ndcg(test, pred, k=50, apply_mean=mean)
    return metrics

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def write_metrics(run_name):
    writer = SummaryWriter(log_dir=f'runs/{run_name}')

    df = pd.read_parquet(f'metrics/{run_name}_val.pqt')
    df = df.apply(mean_confidence_interval)
    df.index = ['mean', 'conf']
    for metric_name, metric_value in df.items():
        writer.add_scalar(f'Val/{metric_name}', metric_value['mean'], 0)

    df = pd.read_parquet(f'metrics/{run_name}_test.pqt')
    df = df.apply(mean_confidence_interval)
    df.index = ['mean', 'conf']
    for metric_name, metric_value in df.items():
        writer.add_scalar(f'Test/{metric_name}', metric_value['mean'], 0)

    writer.close()

