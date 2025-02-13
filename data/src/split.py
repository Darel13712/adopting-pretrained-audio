import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def get_dates(TEST_DATE, train_years=1, test_months=1):
	test_date = pd.to_datetime(TEST_DATE)
	train_date = test_date - pd.DateOffset(years=train_years)
	end_date = test_date + pd.DateOffset(months=test_months)
	return train_date, test_date, end_date

def calc_stats(df):
	track_count = df.groupby('user_id').track_id.count()
	play_count = df.groupby('user_id')['count'].sum()

	return (len(df),
	df.user_id.nunique(),
	df.track_id.nunique(),
	round(track_count.median()),
	round(track_count.min()),
	round(track_count.max()),
	round(play_count.median()),
	round(play_count.min()),
	round(play_count.max())
	)

def get_stats(train, val, hot_test, cold_test):
	res = pd.DataFrame(columns=['Rows', 'Users', 'Items', 'TracksMedian', 'TracksMin', 'TracksMax', 'PlaysMedian', 'PlaysMin', 'PlaysMax'])
	res.loc['Train', :] = calc_stats(train)
	res.loc['Val', :] = calc_stats(val)
	res.loc['HotTest', :] = calc_stats(hot_test)
	res.loc['ColdTest', :] = calc_stats(cold_test)
	return res



compress = lambda x: x.groupby(['user_id', 'track_id']).agg(timestamp=("timestamp", "min"), count=("timestamp", "count")).reset_index()

def date_split(df, TEST_DATE):
	train_date, test_date, end_date = get_dates(TEST_DATE)

	# Select time span
	df = df[(df.timestamp >= train_date) & (df.timestamp < end_date)]
	a = compress(df)

	# Get test data
	te = a[a.timestamp >= test_date]

	# This is done so that the number of plays in train doesn't account for plays that happened after TEST_DATE
	df = df[df.timestamp < test_date]
	tr = compress(df)

	# Filter cold users out
	te = te[te.user_id.isin(tr.user_id.unique())]

	# Split test into hot and cold parts
	test_item_isin_train = te.track_id.isin(tr.track_id.unique())
	cold = te[~test_item_isin_train]
	hot = te[test_item_isin_train]

	# We don't want to put users with cold items in val because there's so little of them
	users_without_cold_items = hot[~hot.user_id.isin(cold.user_id.unique())].user_id.unique()

	# Split users into val and test
	validation_user_ids, test_user_ids = train_test_split(users_without_cold_items, test_size=0.7, random_state=42)
	val = hot[hot.user_id.isin(validation_user_ids)]
	test = hot[~hot.user_id.isin(validation_user_ids)]
	return tr, val, test, cold


TEST_DATE = '2019-02-01'

df = pd.read_parquet('/gpfs/space/projects/music_ca/DATA/music4all/plays.pqt')
train, val, hot_test, cold_test = date_split(df, TEST_DATE)
stats = get_stats(train, val, hot_test, cold_test)
stats.to_csv('stats.csv')

train.to_parquet('train.pqt')
val.to_parquet('val.pqt')
hot_test.to_parquet('hot_test.pqt')
cold_test.to_parquet('cold_test.pqt')






