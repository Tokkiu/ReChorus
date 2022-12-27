import os
import re
import zipfile
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
DATASET = 'ml-1m'  # only support "ml-100k" and "ml-1m" now
RAW_PATH = os.path.join('./', DATASET)

RANDOM_SEED = 0
NEG_ITEMS = 99

# download data if not exists

if not os.path.exists(RAW_PATH):
    subprocess.call('mkdir ' + RAW_PATH, shell=True)
if not os.path.exists(os.path.join(RAW_PATH, DATASET + '.zip')):
    print('Downloading data into ' + RAW_PATH)
    subprocess.call(
        'cd {} && curl -O http://files.grouplens.org/datasets/movielens/{}.zip'
        .format(RAW_PATH, DATASET), shell=True)

with zipfile.ZipFile(os.path.join(RAW_PATH, DATASET + '.zip')) as z:
    if DATASET == 'ml-100k':
        with z.open(os.path.join(DATASET, 'u.data')) as f:
            data_df = pd.read_csv(f, sep="\t", header=None)
        with z.open(os.path.join(DATASET, 'u.item')) as f:
            meta_df = pd.read_csv(f, sep='|', header=None, encoding='ISO-8859-1')
    elif DATASET == 'ml-1m':
        with z.open(os.path.join(DATASET, 'ratings.dat')) as f:
            data_df = pd.read_csv(f, sep='::', header=None, engine='python')
        with z.open(os.path.join(DATASET, 'movies.dat')) as f:
            meta_df = pd.read_csv(f, sep='::', header=None, engine='python', encoding='ISO-8859-1')
data_df.columns = ['user_id', 'item_id', 'label', 'time']
data_df.head()

genres = [
    'i_Action', 'i_Adventure', 'i_Animation', "i_Children's", 'i_Comedy', 'i_Crime',
    'i_Documentary', 'i_Drama', 'i_Fantasy', 'i_Film-Noir', 'i_Horror', 'i_Musical',
    'i_Mystery', 'i_Romance', 'i_Sci-Fi', 'i_Thriller', 'i_War', 'i_Western', 'i_Other'
]
if DATASET == 'ml-100k':
    item_df = meta_df.drop([1, 3, 4], axis=1)
    item_df.columns = ['item_id', 'i_year'] + genres
elif DATASET == 'ml-1m':
    item_df = meta_df.copy()
    item_df.columns = ['item_id', 'title', 'genre']
    # item_df['title'] = item_df['title'].apply(lambda x: x.decode('ISO-8859-1'))
    # item_df['genre'] = item_df['genre'].apply(lambda x: x.decode('ISO-8859-1'))
    genre_dict = dict()
    for g in genres:
        genre_dict[g] = np.zeros(len(item_df), dtype=np.int32)
    item_genre = item_df['genre'].apply(lambda x: x.split('|')).values
    for idx, genre_lst in enumerate(item_genre):
        for g in genre_lst:
            genre_dict['i_' + g][idx] = 1
    for g in genres:
        item_df[g] = genre_dict[g]
    item_df = item_df.drop(columns=['genre'])
item_df.head()

# Only retain users and items with at least 5 associated interactions

# print('Filter before:', len(data_df))
# filter_before = -1
# while filter_before != len(data_df):
#     filter_before = len(data_df)
#     for stage in ['user_id', 'item_id']:
#         val_cnt = data_df[stage].value_counts()
#         cnt_df = pd.DataFrame({stage: val_cnt.index, 'cnt': val_cnt.values})
#         data_df = pd.merge(data_df, cnt_df, on=stage, how='left')
#         data_df = data_df[data_df['cnt'] >= 5].drop(columns=['cnt'])
# print('Filter after:', len(data_df))

item_df = item_df[item_df['item_id'].isin(data_df['item_id'])]  # remove unuseful metadata

n_users = data_df['user_id'].value_counts().size
n_items = data_df['item_id'].value_counts().size
n_clicks = len(data_df)
min_time = data_df['time'].min()
max_time = data_df['time'].max()

time_format = '%Y-%m-%d'

print('# Users:', n_users)
print('# Items:', n_items)
print('# Interactions:', n_clicks)
print('Time Span: {}/{}'.format(
    datetime.utcfromtimestamp(min_time).strftime(time_format),
    datetime.utcfromtimestamp(max_time).strftime(time_format))
)

np.random.seed(RANDOM_SEED)

out_df = data_df[['user_id', 'item_id', 'time']]
out_df = out_df.drop_duplicates(['user_id', 'item_id', 'time'])
out_df.sort_values(by=['time', 'user_id'], kind='mergesort', inplace=True)
out_df = out_df.reset_index(drop=True)
out_df.head()

# reindex (start from 1)

uids = sorted(out_df['user_id'].unique())
user2id = dict(zip(uids, range(1, len(uids) + 1)))
iids = sorted(out_df['item_id'].unique())
item2id = dict(zip(iids, range(1, len(iids) + 1)))

out_df['user_id'] = out_df['user_id'].apply(lambda x: user2id[x])
out_df['item_id'] = out_df['item_id'].apply(lambda x: item2id[x])
out_df.head()

# leave one out spliting

clicked_item_set = dict()
for user_id, seq_df in out_df.groupby('user_id'):
    clicked_item_set[user_id] = set(seq_df['item_id'].values.tolist())


def generate_dev_test(data_df):
    result_dfs = []
    n_items = data_df['item_id'].value_counts().size
    for idx in range(2):
        result_df = data_df.groupby('user_id').tail(1).copy()
        data_df = data_df.drop(result_df.index)
        neg_items = np.random.randint(1, n_items + 1, (len(result_df), NEG_ITEMS))
        for i, uid in enumerate(result_df['user_id'].values):
            user_clicked = clicked_item_set[uid]
            for j in range(len(neg_items[i])):
                while neg_items[i][j] in user_clicked:
                    neg_items[i][j] = np.random.randint(1, n_items + 1)
        result_df['neg_items'] = neg_items.tolist()
        result_dfs.append(result_df)
    return result_dfs, data_df

leave_df = out_df.groupby('user_id').head(1)
data_df = out_df.drop(leave_df.index)

[test_df, dev_df], data_df = generate_dev_test(data_df)
train_df = pd.concat([leave_df, data_df]).sort_index()

len(train_df), len(dev_df), len(test_df)

# save results

train_df.to_csv(os.path.join(RAW_PATH, 'train.csv'), sep='\t', index=False)
dev_df.to_csv(os.path.join(RAW_PATH, 'dev.csv'), sep='\t', index=False)
test_df.to_csv(os.path.join(RAW_PATH, 'test.csv'), sep='\t', index=False)

item_df['item_id'] = item_df['item_id'].apply(lambda x: item2id[x])

if DATASET == 'ml-1m':
    item_df['i_year'] = item_df['title'].apply(lambda x: int(re.match('.+\((\d{4})\)$', x).group(1)))
    item_df = item_df.drop(columns=['title'])
elif DATASET == 'ml-100k':
    item_df['i_year'] = item_df['i_year'].apply(lambda x: int(str(x).split('-')[-1]) if pd.notnull(x) else 0)
seps = [1900, 1940, 1950, 1960, 1970, 1980, 1985] + list(range(1990, int(item_df['i_year'].max() + 2)))
year_dict = {}
for i, sep in enumerate(seps[:-1]):
    for j in range(seps[i], seps[i + 1]):
        year_dict[j] = i + 1
item_df['i_year'] = item_df['i_year'].apply(lambda x: year_dict[x] if x > 0 else 0)

item_df.head()

# save results

item_df.to_csv(os.path.join(RAW_PATH, 'item_meta.csv'), sep='\t', index=False)