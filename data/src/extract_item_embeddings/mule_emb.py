from scooch import Config
from mule.analysis import Analysis
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import pandas as pd


track_ids = pd.read_csv('/gpfs/space/projects/music_ca/DATA/music4all/trackid_sorted.csv', index_col=0)['trackid']
track_paths = [ '/gpfs/space/projects/music_ca/DATA/music4all/wav44k/' + track_id + '.wav' for track_id in track_ids]


mode = "average"
# mode = "timeline"

config = f"/gpfs/helios/home/yanmart/audio_tools/music-audio-representations/supporting_data/configs/mule_embedding_{mode}.yml"


input_file = "/gpfs/helios/home/yanmart/tj.wav"
input_file = "/gpfs/helios/home/yanmart/tjj.wav"
input_file = "/gpfs/helios/home/yanmart/t16k.wav"

def get_embedding(file_path):
	return analysis.analyze(file_path)._data.reshape(-1)


cfg = Config(config)
analysis = Analysis(cfg)
# feat = analysis.analyze(input_file)._data
# if mode == "average":
# 	feat = feat.reshape(-1)

res = []
for track in tqdm(track_paths):
	res.append(get_embedding(track))

res = np.stack(res)
np.save(f'mule.npy', res)