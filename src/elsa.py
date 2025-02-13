import pandas as pd
from scipy.sparse import coo_matrix
# from implicit.als import AlternatingLeastSquares as ALS
import numpy as np
from tqdm import tqdm
from elsa import ELSA
import scipy
from rs_metrics import hitrate, mrr, precision, recall, ndcg
from sklearn.preprocessing import LabelEncoder
from utils import *
from torch.utils.tensorboard import SummaryWriter


train, val, hot_test, cold_test, ue, ie = load_dataset('m4a')

ratings_matrix = coo_matrix((
    np.ones(len(train)),
    (train.user_id, train.item_id)),
shape=(train.user_id.nunique(), len(ie.classes_))
).tocsr()

val_data = coo_matrix((
    np.ones(len(val)),
    (val.user_id, val.item_id)),
shape=(train.user_id.nunique(), len(ie.classes_))
).tocsr()


user_history = train.groupby('user_id')['item_id'].agg(set).to_dict()


items_cnt = ratings_matrix.shape[1]
factors = 768 
num_epochs = 10
batch_size = 128
lr = 0.01

run_name = f'elsa_{factors}_{num_epochs}_{lr}'
writer = SummaryWriter(log_dir='runs/' + run_name)

model = ELSA(n_items=items_cnt, n_dims=factors)
d = model.fit(ratings_matrix, batch_size=batch_size, epochs=num_epochs, validation_data=val_data)
pred = model.predict(ratings_matrix, batch_size).cpu().numpy()

user_recommendations = {}
k = 100
for user_id in tqdm(hot_test.user_id.unique()):
    history = user_history[user_id]
    user_vector = pred[user_id]
    indices = np.argsort(user_vector)[::-1][:k + len(history)]
    recommendations = [idx for idx in indices if idx not in history][:k]
    user_recommendations[user_id] = recommendations

# Convert recommendations to DataFrame
df = dict_to_pandas(user_recommendations)
df.to_parquet(f'preds/{run_name}_hot.pqt')

# Calculate metrics
metrics_test_hot = calc_metrics(hot_test, df, False)
metrics_test_hot.to_parquet(f'metrics/{run_name}_hot_test.pqt')
metrics_test_hot = metrics_test_hot.apply(mean_confidence_interval)
metrics_test_hot.index = ['mean', 'conf']
metrics_test_hot.to_csv(f'metrics/{run_name}_hot_test.csv')
print('Hot test metrics:')
print(metrics_test_hot)

for metric_name, metric_value in metrics_test_hot.items():
    writer.add_scalar(f'Hot test/{metric_name}', metric_value['mean'], 0)
writer.close()

embs = model.get_items_embeddings(as_numpy=True)
np.save(f'model_embeddings/{run_name}_items.npy', embs)


