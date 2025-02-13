# Adopting State-of-the-Art Pretrained Audio Representations for Music Recommender Systems


This repository contains a code that was used to produce results for the paper "Adopting State-of-the-Art Pretrained Audio Representations for Music Recommender Systems".
This is an improved and extended journal version of the paper "Comparative Analysis of Pretrained Audio Representations in Music Recommender Systems".



## Extract Model Embeddings

Code to extract pretrained embeddings is located in `data/extract_item_embeddings/`. 

For each model there are two files:

- `model.py` which reads files and calculates embeddings 
- `model.sh` which is a slurm job file that was used to submit the job to a cluster with a `sbatch model.sh`

Different models use different input sample rates that were precomputed with `ffmpeg`.

We store precomputed embeddings (not provided in this repository due to size) as `.npy` arrays. Therefore our internal index for tracks differs from original `track_id`. To get our index we sorted all `track_id` in the dataset and used index as a new id. The resulting mapping can be found in the `trackid_sorted.csv` 

## Train Test Split
`data/` folder:
Contains actual splits used for the experiment.

`data/src/` folder:
- `0_get_plays_pqt.py` converts Music4All track_ids to our indexes
- `split.py` splits the log file into train/test/validation and compresses a raw log into a user-playcount format

## Model Definitions

- `bert4rec.py` contains model definition for BERT4Rec
- `bert4rec.yaml` contains default parameters for BERT4Rec, most of them are subject to be redefined for a training with `train_bert.sh`
- `model.py` defines `ShallowEmbeddingModel` which is referenced as Shallow Net in the paper
- `dataset.py` defines `InteractionDatasetItems` used to handle dataset.
- `knn.py` generates results for the KNN model
- `contrastive.py` defines the Bimodal net
- `elsa.py` trains ELSA collaborative model to use as part of the Bimodal net training
- `hybrid.py` defines the hybrid model
- `utils.py` helper functions


## Training

Training happens in `train.py` for Shallow Net, `train_contrast.py` for Bimodal, `train_hybrid` for Hybrid and `train_bert.py` for BERT4Rec. We used slurm manager for submitting jobs and to we created `train.sh` and `train_bert.sh` and so on to help with submitting many jobs with different parameters.

## Collecting Results

- `Metrics.ipynb` calculates all the metrics using [rs_metrics](https://github.com/Darel13712/rs_metrics) and produces a report
