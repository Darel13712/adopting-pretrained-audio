import sys
import os
import argparse
import torch
import pandas as pd
import numpy as np
import faiss
import scipy
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from rs_metrics import hitrate, mrr, precision, recall, ndcg

from dataset import InteractionDataset, InteractionDatasetItems
from model import ShallowEmbeddingModel
from utils import *


def hinge_loss(y_pos, y_neg, confidence, dlt=0.2):
    loss = dlt - y_pos + y_neg
    loss = torch.clamp(loss, min=0) * confidence
    return torch.mean(loss)

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

def make_small_train(tr, te, em, num=100000):
    tr = tr.head(num)
    te = te[te.user_id.isin(tr.user_id.unique()) & te.track_id.isin(tr.track_id.unique())]
    le = LabelEncoder()
    tr['user_id'] = le.fit_transform(tr['user_id'])
    te['user_id'] = le.transform(te['user_id'])
    tr['item_id'] = le.fit_transform(tr['track_id'])
    te['item_id'] = le.transform(te['track_id'])
    em = em[le.classes_]
    return tr, te, em

def main():

    parser = argparse.ArgumentParser(description="Train a shallow net")
    
    parser.add_argument("--dataset", required=True, choices=['m4a', 'lfm'], help="m4a / lfm")
    parser.add_argument("--model_name", required=True, help="Name of the pretrained model (musicnn, ...)")
    parser.add_argument("--sample_type", default="item", choices=['item', 'user'], help="user / item. Only item is used in the research")
    parser.add_argument("--neg_samples", type=int, default=20, help="Number of negative samples")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size (default: 10000)")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument("--item_freeze", type=int, choices=[0, 1], default=1, help="Freeze item (0 or 1, default: 1)")
    parser.add_argument("--user_freeze", type=int, choices=[0, 1], default=0, help="Freeze user (0 or 1, default: 0)")
    parser.add_argument("--comment", default="", help="Additional comment")
    parser.add_argument("--user_init", type=int, choices=[0, 1], default=1, help="Initialize user (0 or 1, default: 1)")
    parser.add_argument("--dynamic_item_freeze", type=int, choices=[0, 1], default=0, help="Dynamic item freeze (0 or 1, default: 0)")
    parser.add_argument("--hidden_dim", type=int, default=0, help="Hidden dimension (default: 0, meaning keep original size)")
    parser.add_argument("--use_confidence", type=int, choices=[0, 1], default=0, help="Use confidence (0 or 1, default: 0)")
    parser.add_argument("--l2", type=float, default=0, help="L2 regularization (default: 0)")
    parser.add_argument("--logdir", default=None, help="Log directory")
    parser.add_argument("--last_epoch", type=int, default=0, help="Last epoch (default: 0)")
    parser.add_argument("--shuffle", type=int, choices=[0, 1], default=0, help="Shuffle pretrained embeddings (sanity check)")

    args = parser.parse_args()

    item_freeze = bool(args.item_freeze)
    user_freeze = bool(args.user_freeze)
    user_init = bool(args.user_init)
    dynamic_item_freeze = bool(args.dynamic_item_freeze)
    use_confidence = bool(args.use_confidence)
    shuffle = bool(args.shuffle)

    print(f"Dataset: {args.dataset}")
    print(f"Model Name: {args.model_name}")
    print(f"Sample Type: {args.sample_type}")
    print(f"Negative Samples: {args.neg_samples}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Num Epochs: {args.num_epochs}")
    print(f"Item Freeze: {item_freeze}")
    print(f"User Freeze: {user_freeze}")
    print(f"Comment: {args.comment}")
    print(f"User Init: {user_init}")
    print(f"Dynamic Item Freeze: {dynamic_item_freeze}")
    print(f"Hidden Dimension: {args.hidden_dim}")
    print(f"Use Confidence: {use_confidence}")
    print(f"L2 Regularization: {args.l2}")
    print(f"Logdir: {args.logdir}")
    print(f"Last Epoch: {args.last_epoch}")
    print(f"Shuffle: {shuffle}")

    dataset = args.dataset
    model_name = args.model_name
    sample_type = args.sample_type
    neg_samples = args.neg_samples
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    hidden_dim = args.hidden_dim





    current_time = datetime.now()
    formatted_time = current_time.strftime("%b%d_%H:%M")
    run_name = f"{dataset}-{model_name}-{formatted_time}_{args.comment}"
    run_name = args.logdir or run_name
    writer = SummaryWriter(log_dir='runs/' + run_name)

    args_dict = vars(args)
    formatted_args = "\n".join(f"{key}: {value}" for key, value in args_dict.items())
    writer.add_text("Arguments", formatted_args)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print(device)


    if model_name == "random":
        embs = np.random.rand(56512, hidden_dim if hidden_dim != 0 else 128)
    else:
        embs = np.load(f'/gpfs/space/projects/music_ca/DATA/{"music4all" if dataset == "m4a" else dataset}/embeddings/{model_name}.npy')
    emb_dim_in = embs.shape[1]
    if hidden_dim == 0:
        hidden_dim = emb_dim_in

    # train, val, embs = make_small_train(train, val, embs, 50000)
    if model_name == "random":
        embs = None

    train, val, hot_test, cold_test, ue, ie = load_dataset(dataset)
    mixed_test = pd.concat([hot_test, cold_test], ignore_index=True)
    user_history = train.groupby('user_id')['item_id'].agg(set).to_dict()

    unique_track_id = np.sort(np.unique(np.concatenate((train.track_id.unique(), cold_test.track_id.unique()))))
    if embs is not None:
        embs = embs[unique_track_id] # this essentially converts initial track_ids to item_ids for embs
        if shuffle:
            np.random.shuffle(embs)

    user_embs = np.stack(train.groupby('user_id')['item_id'].apply(lambda items: embs[items].mean(axis=0)).values) if (user_init and embs is not None) else None

    Dataset = InteractionDataset if sample_type == "user" else InteractionDatasetItems

    train_dataset = Dataset(train, neg_samples=neg_samples)
    val_dataset = Dataset(val, neg_samples=neg_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ShallowEmbeddingModel(
        train.user_id.nunique(),
        len(ie.classes_),  # Total number of items including cold test items
        emb_dim_in,
        precomputed_item_embeddings=embs,
        precomputed_user_embeddings=user_embs,
        emb_dim_out=hidden_dim
    )
    model.freeze_item_embs(item_freeze)
    model.freeze_user_embs(user_freeze)

    model.to(device)

    patience_counter = 0
    patience_threshold = 16

    best_val_loss = float('inf')
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
    os.makedirs(f'checkpoints/{model_name}', exist_ok=True)

    if args.last_epoch > 0:
        model, optimizer, loss = load_checkpoint(model, optimizer, f'checkpoints/{model_name}/{run_name}_{args.last_epoch - 1}.pt')
        best_val_loss = torch.load(f'checkpoints/{model_name}/{run_name}_best.pt')['loss']

    for epoch in tqdm(range(args.last_epoch, num_epochs)):
        model.train()
        total_train_loss = 0

        for user, positive_item, confidence in train_loader:
            user = user.repeat_interleave(neg_samples).to(device)
            positive_item = positive_item.repeat_interleave(neg_samples).to(device)
            negative_items = torch.from_numpy(np.random.randint(0, 17052, len(user))).to(device)
            # negative_items = negative_items.view(-1).to(device)
            if not use_confidence:
                confidence = torch.tensor([1]*len(confidence))
            confidence = confidence.repeat_interleave(neg_samples).to(device)
            if use_confidence:
                confidence = (1 + 2 * torch.log(1 + confidence))
            optimizer.zero_grad()

            if sample_type == 'user':
                pos_score = model(user, positive_item)
                neg_scores = model(user, negative_items)
            elif sample_type == 'item': # In this case Dataset samples users instead of items
                pos_score = model(positive_item, user)
                neg_scores = model(negative_items, user)
            loss = hinge_loss(pos_score, neg_scores, confidence)
            if args.l2 > 0:
                l2_loss = sum(torch.sum(param ** 2) for param in model.parameters())
                loss = loss + args.l2 * l2_loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        # save_model(model, f'checkpoints/{model_name}/{run_name}_{epoch}.pt', epoch, optimizer)

        # if epoch % 5 == 0 and epoch > 0:
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for user, positive_item, confidence in val_loader:
                user = user.repeat_interleave(neg_samples).to(device)
                positive_item = positive_item.repeat_interleave(neg_samples).to(device)
                negative_items = torch.from_numpy(np.random.randint(0, 17052, len(user))).to(device)
                # negative_items = negative_items.view(-1).to(device)
                if not use_confidence:
                    confidence = torch.tensor([1]*len(confidence))
                confidence = confidence.repeat_interleave(neg_samples).to(device)
                if use_confidence:
                    confidence = (1 + 2 * torch.log(1 + confidence))
                if sample_type == 'user':
                    pos_score = model(user, positive_item)
                    neg_scores = model(user, negative_items)
                elif sample_type == 'item': # In this case Dataset samples users instead of items
                    pos_score = model(positive_item, user)
                    neg_scores = model(negative_items, user)
                loss = hinge_loss(pos_score, neg_scores, confidence)
                if args.l2 > 0:
                    l2_loss = sum(torch.sum(param ** 2) for param in model.parameters())
                    loss = loss + args.l2 * l2_loss
                total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                print(f'new best at epoch {epoch}')
                best_val_loss = avg_val_loss
                patience_counter = 0
                save_model(model, f'checkpoints/{model_name}/{run_name}_best.pt', epoch, optimizer, best_val_loss)
            else:
                patience_counter += 1
            if patience_counter == patience_threshold:
                print('Applying early stop')
                break

    model, optimizer, loss = load_checkpoint(model, optimizer, f'checkpoints/{model_name}/{run_name}_best.pt')
    user_embs, item_embs = model.extract_embeddings()
    os.makedirs(f'model_embeddings/', exist_ok=True)
    np.save(f'model_embeddings/{run_name}_users.npy', user_embs)
    np.save(f'model_embeddings/{run_name}_items.npy', item_embs)

    index = faiss.IndexFlatIP(item_embs.shape[1])
    index.add(item_embs)

    index_cold = faiss.IndexIDMap(faiss.IndexFlatIP(item_embs.shape[1]))
    index_cold.add_with_ids(item_embs[cold_test.item_id.unique()], cold_test.item_id.unique())


    hot_recommendations = {}
    cold_recommendations = {}
    cold_items = set(cold_test.item_id.unique())
    k = 100

    # all_users = np.unique(np.concatenate((val.user_id.unique(), cold_test.user_id.unique(), hot_test.user_id.unique())))
    all_users = np.unique(np.concatenate((cold_test.user_id.unique(), hot_test.user_id.unique())))
    for user_id in tqdm(all_users):
        history = user_history[user_id]
        user_vector = user_embs[user_id]
        distances, indices = index.search(np.array([user_vector]), k + len(history) + len(cold_items))
        base_recs = [idx for idx in indices[0] if idx not in history]
        hot_recommendations[user_id] = [item_id for item_id in base_recs if item_id not in cold_items][:k]

        distances, indices = index_cold.search(np.array([user_vector]), k + len(history))
        cold_recommendations[user_id] = [idx for idx in indices[0] if idx not in history][:k]


    # df['item_id'] = ie.inverse_transform(df['item_id'])
    # df['user_id'] = ue.inverse_transform(df['user_id'])
    os.makedirs(f'preds/', exist_ok=True)

    # mixed_pred = dict_to_pandas(mixed_recommendations)
    # mixed_pred.to_parquet(f'preds/{run_name}_mixed.pqt')

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

    # metrics_test_mixed = calc_metrics(mixed_test, mixed_pred, False)
    # metrics_test_mixed.to_parquet(f'metrics/{run_name}_mixed_test.pqt')
    # metrics_test_mixed = metrics_test_mixed.apply(mean_confidence_interval)
    # metrics_test_mixed.index = ['mean', 'conf']
    # metrics_test_mixed.to_csv(f'metrics/{run_name}_mixed_test.csv')
    # print('Mixed test metrics:')
    # print(metrics_test_mixed)

    metrics_test_hot = calc_metrics(hot_test, hot_pred, False)
    metrics_test_hot.to_parquet(f'metrics/{run_name}_hot_test.pqt')
    metrics_test_hot = metrics_test_hot.apply(mean_confidence_interval)
    metrics_test_hot.index = ['mean', 'conf']
    metrics_test_hot.to_csv(f'metrics/{run_name}_hot_test.csv')
    print('Hot test metrics:')
    print(metrics_test_hot)

    metrics_test_cold = calc_metrics(cold_test, cold_pred, False, 20)
    metrics_test_cold.to_parquet(f'metrics/{run_name}_cold_test.pqt')
    metrics_test_cold = metrics_test_cold.apply(mean_confidence_interval)
    metrics_test_cold.index = ['mean', 'conf']
    metrics_test_cold.to_csv(f'metrics/{run_name}_cold_test.csv')
    print('Cold test metrics:')
    print(metrics_test_cold)


    # for metric_name, metric_value in metrics_val.items():
    #     writer.add_scalar(f'Val/{metric_name}', metric_value['mean'], 0)

    # for metric_name, metric_value in metrics_test_mixed.items():
    #     writer.add_scalar(f'Mix test/{metric_name}', metric_value['mean'], 0)

    for metric_name, metric_value in metrics_test_hot.items():
        writer.add_scalar(f'Hot test/{metric_name}', metric_value['mean'], 0)

    for metric_name, metric_value in metrics_test_cold.items():
        writer.add_scalar(f'Cold test/{metric_name}', metric_value['mean'], 0)

    writer.close()

if __name__ == '__main__':
    main()
