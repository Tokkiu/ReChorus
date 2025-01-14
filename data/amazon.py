import os
import gzip
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

DATASET = 'Grocery_and_Gourmet_Food'
RAW_PATH = os.path.join('./', DATASET)
DATA_FILE = 'reviews_{}_5.json.gz'.format(DATASET)
META_FILE = 'meta_{}.json.gz'.format(DATASET)

RANDOM_SEED = 0
NEG_ITEMS = 99

# download data if not exists

if not os.path.exists(RAW_PATH):
    subprocess.call('mkdir ' + RAW_PATH, shell=True)
if not os.path.exists(os.path.join(RAW_PATH, DATA_FILE)):
    print('Downloading interaction data into ' + RAW_PATH)
    subprocess.call(
        'cd {} && curl -O http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{}_5.json.gz'
        .format(RAW_PATH, DATASET), shell=True)
if not os.path.exists(os.path.join(RAW_PATH, META_FILE)):
    print('Downloading item metadata into ' + RAW_PATH)
    subprocess.call(
        'cd {} && curl -O http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_{}.json.gz'
        .format(RAW_PATH, DATASET), shell=True)

data_df = get_df(os.path.join(RAW_PATH, DATA_FILE))
data_df.head()

meta_df = get_df(os.path.join(RAW_PATH, META_FILE))
meta_df.head()

# Only retain items that appear in interaction data

useful_meta_df = meta_df[meta_df['asin'].isin(data_df['asin'])].reset_index(drop=True)
all_items = set(useful_meta_df['asin'].values.tolist())

def related_filter(related_dict):
    out_dict = dict()
    if related_dict is not np.nan:
        for r in related_dict:
            out_dict[r] = list(all_items & set(related_dict[r]))
    return out_dict

useful_meta_df['related'] = useful_meta_df['related'].apply(related_filter)

n_users = data_df['reviewerID'].value_counts().size
n_items = data_df['asin'].value_counts().size
n_clicks = len(data_df)
min_time = data_df['unixReviewTime'].min()
max_time = data_df['unixReviewTime'].max()

time_format = '%Y-%m-%d'

print('# Users:', n_users)
print('# Items:', n_items)
print('# Interactions:', n_clicks)
print('Time Span: {}/{}'.format(
    datetime.utcfromtimestamp(min_time).strftime(time_format),
    datetime.utcfromtimestamp(max_time).strftime(time_format))
)

np.random.seed(RANDOM_SEED)

out_df = data_df.rename(columns={'asin': 'item_id', 'reviewerID': 'user_id', 'unixReviewTime': 'time'})
out_df = out_df[['user_id', 'item_id', 'time']]
out_df = out_df.drop_duplicates(['user_id', 'item_id', 'time'])
out_df = out_df.sort_values(by=['time', 'user_id'], kind='mergesort').reset_index(drop=True)
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

# level-2 category

l2_cate_lst = list()
for cate_lst in useful_meta_df['categories']:
    l2_cate_lst.append(cate_lst[0][2] if len(cate_lst[0]) > 2 else np.nan)
useful_meta_df['l2_category'] = l2_cate_lst
l2_cates = sorted(useful_meta_df['l2_category'].dropna().unique())
l2_dict = dict(zip(l2_cates, range(1, len(l2_cates) + 1)))
useful_meta_df['l2_category'] = useful_meta_df['l2_category'].apply(lambda x: l2_dict[x] if x == x else 0)

item_meta_data = dict()
for idx in range(len(useful_meta_df)):
    info = useful_meta_df.iloc[idx]['related']
    item_meta_data[idx] = {
        'item_id': item2id[useful_meta_df.iloc[idx]['asin']],
        'i_category': useful_meta_df.iloc[idx]['l2_category'],
        'r_complement': list(map(lambda x: item2id[x], info['also_bought'])) if 'also_bought' in info else [],
        'r_substitute': list(map(lambda x: item2id[x], info['also_viewed'])) if 'also_viewed' in info else [],
    }

item_meta_df = pd.DataFrame.from_dict(item_meta_data, orient='index')
item_meta_df = item_meta_df[['item_id', 'i_category', 'r_complement', 'r_substitute']]
item_meta_df.head()

# save results

item_meta_df.to_csv(os.path.join(RAW_PATH, 'item_meta.csv'), sep='\t', index=False)