import argparse
import torch, librosa
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm
from muq import MuQ, MuQMuLan



def read_mono(path, target_length=720000):
    audio = sf.read(path, stop=target_length)[0]
    if len(audio.shape) == 2:
        audio = audio.mean(axis=1)
    current_length = audio.shape[0]
    
    if current_length < target_length:
        silence_length = target_length - current_length
        silence = np.zeros(silence_length)
        audio = np.concatenate((audio, silence))
    return audio

def load_batch(track_paths, batch_size, index):
    tracks = track_paths[index * batch_size: (index + 1) * batch_size]
    embs = np.stack([read_mono(track) for track in tracks])
    return embs

def embed_batch(model, batch):
    with torch.no_grad():
        batch = torch.from_numpy(batch).float().cuda()
        output = model(batch, output_hidden_states=False)
        output = output.last_hidden_state.mean(axis=1).cpu().numpy()
        return output

def main():

    parser = argparse.ArgumentParser(description="Get Muq embeddings")
    parser.add_argument("--checkpoint", required=True, choices=['msd', 'mulan'], help="msd / mulan")
    args = parser.parse_args()



    track_ids = pd.read_csv('/gpfs/space/projects/music_ca/DATA/music4all/trackid_sorted.csv', index_col=0)['trackid']
    track_paths = [ '/gpfs/space/projects/music_ca/DATA/music4all/wav24k/' + track_id + '.wav' for track_id in track_ids]

    batch_size = 1
    device = 'cuda'

    if args.checkpoint == 'msd':
        muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    elif args.checkpoint == 'mulan':
        muq = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")

    muq = muq.to(device).eval()



    res = []
    for i in tqdm(range(int(len(track_paths) / batch_size))):
        b = load_batch(track_paths, batch_size, i)
        reps = embed_batch(muq, b)
        res.append(reps)

    res = np.concatenate(res)

    np.save(f'embeddings/muq_{args.checkpoint}.npy', e)
