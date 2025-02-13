import sys
import os
import argparse
import torch
import pandas as pd
import numpy as np
import faiss
import scipy
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from rs_metrics import hitrate, mrr, precision, recall, ndcg

from dataset import SparseMatrixDataset
from hybrid import *
from utils import *


def save_model(model, path, epoch, optimizer, best_val_loss=None):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': best_val_loss,
    }, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch}, loss {loss})")
    return model, optimizer, loss

def predict_by_scores(pred, test, user_history, k=100):
    user_recommendations = {}
    for user_id in tqdm(test.user_id.unique()):
        history = user_history[user_id]
        user_vector = pred[user_id]
        indices = np.argsort(user_vector)[::-1]
        recommendations = [idx for idx in indices if idx not in history][:k]
        user_recommendations[user_id] = recommendations
    df = dict_to_pandas(user_recommendations)
    return df


def main():

    parser = argparse.ArgumentParser(description="Train a hybrid model")
    
    parser.add_argument("--dataset", required=True, choices=['m4a', 'lfm'], help="m4a / lfm")
    parser.add_argument("--model_name", required=True, help="Name of the pretrained model (musicnn, ...)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default: 10000)")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs (default: 100)")
    parser.add_argument("--comment", default="", help="Additional comment")
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden dimension")
    parser.add_argument("--logdir", default=None, help="Log directory")
    parser.add_argument("--last_epoch", type=int, default=0, help="Last epoch (default: 0)")
    parser.add_argument("--shuffle", type=int, choices=[0, 1], default=0, help="Shuffle pretrained embeddings (sanity check)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--optimizer", default='adam', choices=['adam', 'nadam'], help="adam / nadam")

    args = parser.parse_args()

    shuffle = bool(args.shuffle)

    print(f"Dataset: {args.dataset}")
    print(f"Model Name: {args.model_name}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Num Epochs: {args.num_epochs}")
    print(f"Comment: {args.comment}")
    print(f"Hidden Dimension: {args.hidden_dim}")
    print(f"Logdir: {args.logdir}")
    print(f"Last Epoch: {args.last_epoch}")
    print(f"Shuffle: {shuffle}")

    dataset = args.dataset
    model_name = args.model_name
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    hidden_dim = args.hidden_dim



    current_time = datetime.now()
    formatted_time = current_time.strftime("%b%d_%H:%M")
    run_name = f"hybrid-{dataset}-{model_name}-{formatted_time}_{args.comment}"
    run_name = args.logdir or run_name
    writer = SummaryWriter(log_dir='runs/' + run_name)

    args_dict = vars(args)
    formatted_args = "\n".join(f"{key}: {value}" for key, value in args_dict.items())
    writer.add_text("Arguments", formatted_args)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print(device)



    embs = np.load(f'/gpfs/space/projects/music_ca/DATA/{"music4all" if dataset == "m4a" else dataset}/embeddings/{model_name}.npy')
    emb_dim_in = embs.shape[1]

    train, val, hot_test, cold_test, ue, ie = load_dataset(dataset)
    user_history = train.groupby('user_id')['item_id'].agg(set).to_dict()

    unique_track_id = np.sort(np.unique(np.concatenate((train.track_id.unique(), cold_test.track_id.unique()))))
    embs = embs[unique_track_id] # this essentially converts initial track_ids to item_ids for embs
    if shuffle:
        np.random.shuffle(embs)


    ratings_matrix = coo_matrix((
        np.ones(len(train)),
        (train.user_id, train.item_id)),
    shape=(train.user_id.nunique(), len(ie.classes_))
    ).tocsr()

    matrix_dataset = SparseMatrixDataset(ratings_matrix)

    train_loader = DataLoader(
        matrix_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: torch.vstack(x),
    )

    val_matrix = coo_matrix((
        np.ones(len(val)),
        (val.user_id, val.item_id)),
    shape=(train.user_id.nunique(), len(ie.classes_))
    ).tocsr()

    non_empty_rows = val_matrix.getnnz(axis=1) > 0
    val_matrix = val_matrix.tocsr()[non_empty_rows]

    val_dataset = SparseMatrixDataset(val_matrix)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: torch.vstack(x),
    )

    model = ELSAHybridModel(
        embs,
        hidden_dim,
    )

    model.to(device)

    patience_counter = 0
    patience_threshold = 16

    criterion = NMSELoss()

    best_loss = float('inf')
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'nadam':
        optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr)

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
    os.makedirs(f'checkpoints/{model_name}', exist_ok=True)


    if args.last_epoch > 0:
        model, optimizer, loss = load_checkpoint(model, optimizer, f'checkpoints/{model_name}/{run_name}_{args.last_epoch - 1}.pt')
        best_loss = torch.load(f'checkpoints/{model_name}/{run_name}_best.pt')['loss']

    last_epoch = args.last_epoch

    for epoch in tqdm(range(last_epoch, num_epochs)):
        model.train()
        total_train_loss = 0

        for rows in train_loader:
            optimizer.zero_grad()

            output = model(rows)
            loss = criterion(output, rows)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(avg_train_loss)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        # save_model(model, f'checkpoints/{model_name}/{run_name}_{epoch}.pt', epoch, optimizer)

        # if epoch % 5 == 0 and epoch > 0:
        #     res = []
        #     with torch.no_grad():
        #         for rows in train_loader:
        #             output = model(rows)
        #             res.append(output)

        #         res = torch.vstack(res).cpu().numpy()

        #     res = predict_by_scores(res, val, user_history, k=100)

        #     metrics_val = calc_metrics(val, res, False)
        #     metrics_val = metrics_val.apply(mean_confidence_interval)
        #     metrics_val.index = ['mean', 'conf']
        #     print('Val metrics:')
        #     print(metrics_val)

        #     for metric_name, metric_value in metrics_val.items():
        #         writer.add_scalar(f'Val/{metric_name}', metric_value['mean'], epoch)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for rows in val_loader:
                output = model(rows)
                loss = criterion(output, rows)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_loss:
            print(f'new best at epoch {epoch}')
            best_loss = avg_val_loss
            patience_counter = 0
            save_model(model, f'checkpoints/{model_name}/{run_name}_best.pt', epoch, optimizer, best_loss)
        else:
            patience_counter += 1
        if patience_counter == patience_threshold:
            print('Applying early stop')
            break

    model, optimizer, loss = load_checkpoint(model, optimizer, f'checkpoints/{model_name}/{run_name}_best.pt')
    os.makedirs(f'preds/', exist_ok=True)
    os.makedirs(f'model_embeddings/', exist_ok=True)

    # res = []
    # with torch.no_grad():
    #     for rows in train_loader:
    #         output = model(rows)
    #         res.append(output)

    #     res = torch.vstack(res).cpu().numpy()


    # df = predict_by_scores(res, hot_test, user_history)
    # df.to_parquet(f'preds/{run_name}_hot.pqt')


    # cold = predict_by_scores(res, cold_test, user_history)
    # cold.to_parquet(f'preds/{run_name}_cold.pqt')

    # metrics_test_hot = calc_metrics(hot_test, df, False)
    # # metrics_test_hot.to_parquet(f'metrics/{run_name}_hot_test.pqt')
    # metrics_test_hot = metrics_test_hot.apply(mean_confidence_interval)
    # metrics_test_hot.index = ['mean', 'conf']
    # metrics_test_hot.to_csv(f'metrics/{run_name}_hot_test.csv')
    # print('Hot test metrics scores:')
    # print(metrics_test_hot)

    # # if typ == 'content':
    # metrics_test_cold = calc_metrics(cold_test, cold, False, 20)
    # # metrics_test_cold.to_parquet(f'metrics/{run_name}_cold_test.pqt')
    # metrics_test_cold = metrics_test_cold.apply(mean_confidence_interval)
    # metrics_test_cold.index = ['mean', 'conf']
    # metrics_test_cold.to_csv(f'metrics/{run_name}_cold_test.csv')
    # print('Cold test metrics scores:')
    # print(metrics_test_cold)

    # for metric_name, metric_value in metrics_test_hot.items():
    #     writer.add_scalar(f'Hot test/{metric_name}', metric_value['mean'], 0)

    # for metric_name, metric_value in metrics_test_cold.items():
    #     writer.add_scalar(f'Cold test/{metric_name}', metric_value['mean'], 0)


    item_embs = model.extract_embeddings()
    np.save(f'model_embeddings/{run_name}_content.npy', item_embs)

    # writer.close()

    
    # writer = SummaryWriter(log_dir='runs/' + run_name + "_faiss")


    index = faiss.IndexFlatIP(item_embs.shape[1])
    index.add(item_embs)

    index_cold = faiss.IndexIDMap(faiss.IndexFlatIP(item_embs.shape[1]))
    index_cold.add_with_ids(item_embs[cold_test.item_id.unique()], cold_test.item_id.unique())

    hot_recommendations = {}
    cold_recommendations = {}
    cold_items = set(cold_test.item_id.unique())
    k = 100

    user_embs = np.stack(train.groupby('user_id')['item_id'].apply(lambda items: item_embs[items].mean(axis=0)).values)
    all_users = np.unique(np.concatenate((val.user_id.unique(), cold_test.user_id.unique(), hot_test.user_id.unique())))
    # all_users = np.unique(np.concatenate((cold_test.user_id.unique(), hot_test.user_id.unique())))
    for user_id in tqdm(all_users):
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

    metrics_val = calc_metrics(val, hot_pred, False)
    metrics_val.to_parquet(f'metrics/{run_name}_val.pqt')
    metrics_val = metrics_val.apply(mean_confidence_interval)
    metrics_val.index = ['mean', 'conf']
    metrics_val.to_csv(f'metrics/{run_name}_val.csv')
    print('Val metrics:')
    print(metrics_val)

    # metrics_test_mixed = calc_metrics(mixed_test, mixed_pred, False)
    # metrics_test_mixed.to_parquet(f'metrics/{run_name}_mixed_test.pqt')
    # metrics_test_mixed = metrics_test_mixed.apply(mean_confidence_interval)
    # metrics_test_mixed.index = ['mean', 'conf']
    # metrics_test_mixed.to_csv(f'metrics/{run_name}_mixed_test.csv')
    # print('Mixed test metrics:')
    # print(metrics_test_mixed)

    metrics_test_hot = calc_metrics(hot_test, hot_pred, False)
    # metrics_test_hot.to_parquet(f'metrics/{run_name}_hot_test.pqt')
    metrics_test_hot = metrics_test_hot.apply(mean_confidence_interval)
    metrics_test_hot.index = ['mean', 'conf']
    metrics_test_hot.to_csv(f'metrics/{run_name}_hot_test.csv')
    print('Hot test metrics:')
    print(metrics_test_hot)

    # if typ == 'content':
    metrics_test_cold = calc_metrics(cold_test, cold_pred, False, 20)
    # metrics_test_cold.to_parquet(f'metrics/{run_name}_cold_test.pqt')
    metrics_test_cold = metrics_test_cold.apply(mean_confidence_interval)
    metrics_test_cold.index = ['mean', 'conf']
    metrics_test_cold.to_csv(f'metrics/{run_name}_cold_test.csv')
    print('Cold test metrics:')
    print(metrics_test_cold)


    for metric_name, metric_value in metrics_val.items():
        writer.add_scalar(f'Val/{metric_name}', metric_value['mean'], 0)

    # for metric_name, metric_value in metrics_test_mixed.items():
    #     writer.add_scalar(f'Mix test/{metric_name}', metric_value['mean'], 0)

    for metric_name, metric_value in metrics_test_hot.items():
        writer.add_scalar(f'Hot test/{metric_name}', metric_value['mean'], 0)

    for metric_name, metric_value in metrics_test_cold.items():
        writer.add_scalar(f'Cold test/{metric_name}', metric_value['mean'], 0)

    writer.close()

if __name__ == '__main__':
    main()
