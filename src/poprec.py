import pandas as pd
import numpy as np
from tqdm import tqdm
from rs_metrics import hitrate, mrr, precision, recall, ndcg
from sklearn.preprocessing import LabelEncoder
from utils import *
from torch.utils.tensorboard import SummaryWriter
import os


# Load dataset
train, val, hot_test, cold_test, ue, ie = load_dataset('m4a')

# Create user history dictionary for filtering already seen items
user_history = train.groupby('user_id')['item_id'].agg(set).to_dict()

# Calculate item popularity (number of interactions per item)
item_popularity = train.groupby('item_id')['user_id'].count().sort_values(ascending=False)


# Setup run configuration
run_name = 'popularity_recommender'

# Create necessary directories
os.makedirs('preds', exist_ok=True)
os.makedirs('metrics', exist_ok=True)
os.makedirs('model_embeddings', exist_ok=True)
os.makedirs('runs', exist_ok=True)

writer = SummaryWriter(log_dir='runs/' + run_name)

# Generate recommendations for hot test users
user_recommendations = {}
k = 100

print("Generating recommendations for hot test users...")
for user_id in tqdm(hot_test.user_id.unique()):
    history = user_history.get(user_id, set())

    # Get top k+len(history) popular items to ensure we have k unseen items
    top_items = item_popularity.index[:k + len(history)]

    # Filter out items the user has already seen
    recommendations = [item_id for item_id in top_items if item_id not in history][:k]

    user_recommendations[user_id] = recommendations

# Convert recommendations to DataFrame and save
print("Saving predictions...")
df = dict_to_pandas(user_recommendations)
df.to_parquet(f'preds/{run_name}_hot.pqt')

# Calculate metrics for hot test
print("Calculating metrics...")
metrics_test_hot = calc_metrics(hot_test, df, False)
metrics_test_hot.to_parquet(f'metrics/{run_name}_hot_test.pqt')
metrics_test_hot = metrics_test_hot.apply(mean_confidence_interval)
metrics_test_hot.index = ['mean', 'conf']
metrics_test_hot.to_csv(f'metrics/{run_name}_hot_test.csv')

print('Hot test metrics:')
print(metrics_test_hot)

# Log metrics to TensorBoard
for metric_name, metric_value in metrics_test_hot.items():
    writer.add_scalar(f'Hot test/{metric_name}', metric_value['mean'], 0)
writer.close()
