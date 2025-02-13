import sys
import os
import argparse
import torch
import pandas as pd
import numpy as np
import faiss
import scipy
import torch.nn.functional as F
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from datetime import datetime
from rs_metrics import hitrate, mrr, precision, recall, ndcg

from bert4rec import BERT4Rec
from utils import *

class LMDataset(Dataset):

    def __init__(self, df, max_length=128, num_negatives=None, full_negative_sampling=True,
                 user_col='user_id', item_col='item_id', time_col='timestamp'):

        self.max_length = max_length
        self.num_negatives = num_negatives
        self.full_negative_sampling = full_negative_sampling
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col

        self.data = df.sort_values(time_col).groupby(user_col)[item_col].agg(list).to_dict()
        self.user_ids = list(self.data.keys())

        if num_negatives:
            self.all_items = df[item_col].unique()

    def __len__(self):

        return len(self.data)

    def sample_negatives(self, item_sequence):

        negatives = self.all_items[~np.isin(self.all_items, item_sequence)]
        if self.full_negative_sampling:
            negatives = np.random.choice(
                negatives, size=self.num_negatives * (len(item_sequence) - 1), replace=True)
            negatives = negatives.reshape(len(item_sequence) - 1, self.num_negatives)
        else:
            negatives = np.random.choice(negatives, size=self.num_negatives, replace=False)

        return negatives

class MaskedLMDataset(LMDataset):

    def __init__(self, df, max_length=128,
                 num_negatives=None, full_negative_sampling=True,
                 mlm_probability=0.2,
                 masking_value=1, ignore_value=-100,
                 force_last_item_masking_prob=0,
                 user_col='user_id', item_col='item_id',
                 time_col='timestamp'):

        super().__init__(df, max_length, num_negatives, full_negative_sampling,
                         user_col, item_col, time_col)

        self.mlm_probability = mlm_probability
        self.masking_value = masking_value
        self.ignore_value = ignore_value
        self.force_last_item_masking_prob = force_last_item_masking_prob

    def __getitem__(self, idx):

        item_sequence = self.data[self.user_ids[idx]]

        if len(item_sequence) > self.max_length:
            item_sequence = item_sequence[-self.max_length:]

        input_ids = np.array(item_sequence)
        mask = np.random.rand(len(item_sequence)) < self.mlm_probability
        input_ids[mask] = self.masking_value
        if self.force_last_item_masking_prob > 0:
            if np.random.rand() < self.force_last_item_masking_prob:
                input_ids[-1] = self.masking_value

        labels = np.array(item_sequence)
        labels[input_ids != self.masking_value] = self.ignore_value

        if self.num_negatives:
            negatives = self.sample_negatives(item_sequence)
            return {'input_ids': input_ids, 'labels': labels, 'negatives': negatives}

        return {'input_ids': input_ids, 'labels': labels}


class MaskedLMPredictionDataset(LMDataset):

    def __init__(self, df, max_length=128, masking_value=1,
                 validation_mode=False,
                 user_col='user_id', item_col='item_id',
                 time_col='timestamp'):

        super().__init__(df, max_length=max_length, num_negatives=None,
                         user_col=user_col, item_col=item_col, time_col=time_col)

        self.masking_value = masking_value
        self.validation_mode = validation_mode

    def __getitem__(self, idx):

        user_id = self.user_ids[idx]
        item_sequence = self.data[user_id]

        if self.validation_mode:
            target = item_sequence[-1]
            input_ids = item_sequence[-self.max_length:-1]
            item_sequence = item_sequence[:-1]
        else:
            input_ids = item_sequence[-self.max_length + 1:]

        input_ids += [self.masking_value]

        if self.validation_mode:
            return {'input_ids': input_ids, 'user_id': user_id,
                    'full_history': item_sequence, 'target': target}
        else:
            return {'input_ids': input_ids, 'user_id': user_id,
                    'full_history': item_sequence}

class PaddingCollateFn:

    def __init__(self, padding_value=0, labels_padding_value=-100):

        self.padding_value = padding_value
        self.labels_padding_value = labels_padding_value

    def __call__(self, batch):

        collated_batch = {}

        for key in batch[0].keys():

            if np.isscalar(batch[0][key]):
                collated_batch[key] = torch.tensor([example[key] for example in batch])
                continue

            if key == 'labels':
                padding_value = self.labels_padding_value
            else:
                padding_value = self.padding_value
            values = [torch.tensor(example[key]) for example in batch]
            collated_batch[key] = pad_sequence(values, batch_first=True,
                                               padding_value=padding_value)

        if 'input_ids' in collated_batch:
            attention_mask = collated_batch['input_ids'] != self.padding_value
            collated_batch['attention_mask'] = attention_mask.to(dtype=torch.float32)

        return collated_batch

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

def add_time_idx(df, user_col='user_id', timestamp_col='timestamp', sort=True):
    """Add time index to interactions dataframe."""

    if sort:
        df = df.sort_values([user_col, timestamp_col])

    df['time_idx'] = df.groupby(user_col).cumcount()
    df['time_idx_reversed'] = df.groupby(user_col).cumcount(ascending=False)

    return df


def main():

    parser = argparse.ArgumentParser(description="Train BERT4Rec")
    
    parser.add_argument("--dataset", required=True, choices=['m4a', 'lfm'], help="m4a / lfm")
    parser.add_argument("--model_name", required=True, help="Name of the pretrained model (musicnn, ...)")
    parser.add_argument("--max_seq_len", type=int, default=300, help="Maximum sequence length for user history (most recent items, default: 300)")
    # parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs (default: 200)")
    parser.add_argument("--item_freeze", type=int, choices=[0, 1], default=1, help="Freeze item (0 or 1, default: 1)")
    parser.add_argument("--comment", default="", help="Additional comment")
    parser.add_argument("--hidden_dim", type=int, default=0, help="Hidden dimension (default: 0, meaning keep original size)")
    parser.add_argument("--logdir", default=None, help="Log directory")
    parser.add_argument("--last_epoch", type=int, default=0, help="Last epoch (default: 0)")
    parser.add_argument("--shuffle", type=int, choices=[0, 1], default=0, help="Shuffle pretrained embeddings (sanity check)")
 
    args = parser.parse_args()

    item_freeze = bool(args.item_freeze)
    max_seq_len = args.max_seq_len
    model_name = args.model_name
    hidden_dim = args.hidden_dim
    shuffle = bool(args.shuffle)

    print(f"Dataset: {args.dataset}")
    print(f"Model Name: {args.model_name}")
    print(f"Max seq len: {args.max_seq_len}")
    # print(f"Batch Size: {args.batch_size}")
    print(f"Num Epochs: {args.num_epochs}")
    print(f"Item Freeze: {item_freeze}")
    print(f"Comment: {args.comment}")
    print(f"Hidden Dimension: {args.hidden_dim}")
    print(f"Logdir: {args.logdir}")
    print(f"Last Epoch: {args.last_epoch}")
    print(f"Shuffle: {shuffle}")



    current_time = datetime.now()
    formatted_time = current_time.strftime("%b%d_%H:%M")
    run_name = f"bert_{model_name}_{args.dataset}-{formatted_time}_{hidden_dim}_{max_seq_len}_{args.comment}"
    run_name = args.logdir or run_name
    writer = SummaryWriter(log_dir='runs/' + run_name)

    args_dict = vars(args)
    formatted_args = "\n".join(f"{key}: {value}" for key, value in args_dict.items())
    writer.add_text("Arguments", formatted_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if model_name != "random":
        embs = np.load(f'/gpfs/space/projects/music_ca/DATA/{"music4all" if args.dataset == "m4a" else args.dataset}/embeddings/{model_name}.npy')
        hidden_dim = embs.shape[1]
    else:
        embs = None

    # if hidden_dim == 0:
    #     hidden_dim = emb_dim_in

    # train, val, embs = make_small_train(train, val, embs, 50000)

    train, val, hot_test, cold_test, ue, ie = load_dataset(args.dataset)
    mixed_test = pd.concat([hot_test, cold_test], ignore_index=True)
    user_history = train.groupby('user_id')['item_id'].agg(set).to_dict()

    unique_track_id = np.sort(np.unique(np.concatenate((train.track_id.unique(), cold_test.track_id.unique()))))
    if embs is not None:
        embs = embs[unique_track_id] # this essentially converts initial track_ids to item_ids for embs
        if shuffle:
            np.random.shuffle(embs)
        special_embs = np.random.normal(loc=0.0, scale=0.02, size=(2, hidden_dim))
        embs = np.concatenate([embs, special_embs], axis=0)



    # vc = val.groupby('user_id')['item_id'].count()
    # val = val[val['user_id'].isin(vc[vc >= 10].index)]
    pred_users = np.unique(np.concatenate((val.user_id.unique(), cold_test.user_id.unique(), hot_test.user_id.unique())))
    pred = train.loc[train.user_id.isin(pred_users)]


    item_count = len(unique_track_id) + 2


    train_dataset = MaskedLMDataset(train, masking_value=item_count-2, max_length=max_seq_len, mlm_probability=0.2, force_last_item_masking_prob=0)
    eval_dataset = MaskedLMDataset(val, masking_value=item_count-2, max_length=max_seq_len, mlm_probability=0.2, force_last_item_masking_prob=0)
    pred_dataset = MaskedLMPredictionDataset(pred, masking_value=item_count-2, max_length=max_seq_len, validation_mode=False)


    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True, num_workers=2,
                              collate_fn=PaddingCollateFn(padding_value=item_count-1))
    eval_loader = DataLoader(eval_dataset, batch_size=128,
                             shuffle=False, num_workers=2,
                             collate_fn=PaddingCollateFn(padding_value=item_count-1))
    pred_loader = DataLoader(pred_dataset, batch_size=128,
                             shuffle=False, num_workers=2,
                             collate_fn=PaddingCollateFn(padding_value=item_count-1))


    model_params = {
        'vocab_size': 2,
        'max_position_embeddings': max_seq_len,
        'hidden_size': hidden_dim,
        'num_hidden_layers': 2,
        'num_attention_heads': 2,
        'intermediate_size': 256
    }
    model = BERT4Rec(vocab_size=item_count, add_head=True, precomputed_item_embeddings=embs, padding_idx=item_count-1,
                     bert_config=model_params)

    model.freeze_item_embs(item_freeze)



    model.to(device)

    patience_counter = 0
    patience_threshold = 16

    best_val_loss = float('inf')
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    os.makedirs(f'checkpoints/{model_name}', exist_ok=True)

    if args.last_epoch > 0:
        model, optimizer, loss = load_checkpoint(model, optimizer, f'checkpoints/{model_name}/{run_name}_{last_epoch - 1}.pt')
        best_val_loss = torch.load(f'checkpoints/{model_name}/{run_name}_best.pt')['loss']

    for epoch in tqdm(range(args.last_epoch, args.num_epochs)):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            input_ids, labels, attention_mask = batch['input_ids'].to(device), batch['labels'].to(device), batch['attention_mask'].to(device)
            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

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
            for batch in eval_loader:
                input_ids, labels, attention_mask = batch['input_ids'].to(device), batch['labels'].to(device), batch['attention_mask'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(eval_loader)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                print(f'New best at epoch {epoch}')
                best_val_loss = avg_val_loss
                patience_counter = 0
                save_model(model, f'checkpoints/{model_name}/{run_name}_best.pt', epoch, optimizer, best_val_loss)
            else:
                patience_counter += 1
            if patience_counter >= patience_threshold:
                print('Applying early stop')
                break

    model, optimizer, loss = load_checkpoint(model, optimizer, f'checkpoints/{model_name}/{run_name}_best.pt')
    # user_embs, item_embs = model.extract_embeddings()


    # mixed_recommendations = {}
    hot_recommendations = {}
    cold_recommendations = {}
    cold_items = set(cold_test.item_id.unique())
    k = 100

    with torch.no_grad():
        model.eval()
        for batch in tqdm(pred_loader):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            user_ids = batch['user_id']

            outputs = model(input_ids, attention_mask)

            seq_lengths = attention_mask.sum(dim=1).long()

            last_item_logits = torch.stack([outputs[i, seq_lengths[i] - 1, :] for i in range(len(seq_lengths))])
            last_item_logits = last_item_logits[:, :-2] # remove mask and padding tokens
            scores, preds = torch.sort(last_item_logits, descending=True)
            preds = preds.cpu().numpy()

            for user_id, item_ids in zip(user_ids, preds):
                user_id = user_id.item()
                history = user_history[user_id]

                base_recs = [item_id for item_id in item_ids if item_id not in history]

                # mixed_recommendations[user_id] = base_recs[:k]
                hot_recommendations[user_id] = [item_id for item_id in base_recs if item_id not in cold_items][:k]
                cold_recommendations[user_id] = [item_id for item_id in base_recs if item_id in cold_items][:k]




    os.makedirs(f'preds/', exist_ok=True)
    # df['item_id'] = ie.inverse_transform(df['item_id'])
    # df['user_id'] = ue.inverse_transform(df['user_id'])

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
