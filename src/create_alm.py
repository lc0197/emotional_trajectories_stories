import os
import shutil
from tqdm import tqdm
from shutil import rmtree

import requests
import tarfile
import pandas as pd
from glob import glob

from config import RES_DIR, DATA_DIR

downloaded_dir = os.path.join(DATA_DIR, 'downloaded', 'alm')
os.makedirs(downloaded_dir, exist_ok=True)

authors = ['Grimms', 'HCAndersen', 'Potter']
potter_gz = "http://people.rc.rit.edu/~coagla/affectdata/Potter.tar.gz"
andersen_gz = "http://people.rc.rit.edu/~coagla/affectdata/HCAndersen.tar.gz"
grimm_gz = "http://people.rc.rit.edu/~coagla/affectdata/Grimms.tar.gz"
gzs = [potter_gz, andersen_gz, grimm_gz]

def download(url, target_file):
    r = requests.get(url, allow_redirects=True)
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    open(target_file, 'wb').write(r.content)

def emmood_to_text(emmood_file):
    with open(emmood_file, 'r') as f:
        emmood_rows = [l.replace('\n', '') for l in f.readlines()]
    return [row[row.rfind('\t')+1:] for row in emmood_rows]

print('Downloading original data...')
for gz in tqdm(gzs):
    target_file = os.path.join(downloaded_dir, gz.split("/")[-1])
    if not os.path.exists(target_file):
        download(gz, target_file)
    tar = tarfile.open(target_file)
    tar.extractall(downloaded_dir)
    tar.close()


tmp_dir = os.path.join(DATA_DIR, '.temp')
os.makedirs(tmp_dir, exist_ok=True)
to_remove_df = pd.read_csv(os.path.join(RES_DIR, 'remove_sentences_from_alm.csv'))

print('Processing downloaded files...')
for author in authors:
    print(f'   ...for {author}')
    emmood_files = glob(os.path.join(downloaded_dir, author, 'emmood', '*.emmood'))
    for emmood_file in tqdm(emmood_files):
        try:
            text = emmood_to_text(emmood_file)
            story_name = os.path.basename(emmood_file).replace('.emmood','')
            # remove sentences?
            to_remove = to_remove_df[to_remove_df.story==story_name]
            if len(to_remove) > 0:
                text = [t for t in text if not t in to_remove.sentence.values]

            df = pd.DataFrame({'author':[author]*len(text), 'story':[story_name]*len(text), 'text':text})
            df.to_csv(os.path.join(tmp_dir, f'{story_name}.csv'), index=False)
        # fails for two tales
        except:
            pass

# partition the stories
target_dir = os.path.join(DATA_DIR, 'tales_va')
os.makedirs(target_dir, exist_ok=True)

partition_df = pd.read_csv(os.path.join(RES_DIR, 'partitioning.csv'))
partition_lists = {'train':[], 'val':[], 'test':[]}

gold_standard_df = pd.read_csv(os.path.join(RES_DIR, 'gold_standard.csv'))

print('Creating partitions...')
for _,row in partition_df.iterrows():
    story = row['story']
    author = row['author']
    partition = row['partition']
    story_text = pd.read_csv(os.path.join(tmp_dir, f'{story}.csv')).text.values
    story_gs = gold_standard_df[gold_standard_df.story==story].iloc[:,-2:].values
    story_ids = gold_standard_df[gold_standard_df.story==story]['ID'].values
    assert story_gs.shape[0] == len(story_text)
    story_len = len(story_text)
    story_df = pd.DataFrame({'ID':story_ids,
                             'story':[story]*story_len,
                             'author':[author]*story_len,
                             'text':story_text,
                             'V_EWE':list(story_gs[:,0]),
                             'A_EWE':list(story_gs[:,1])})
    partition_lists[partition].append(story_df)

for partition, dfs in partition_lists.items():
    df = pd.concat(dfs)
    df.to_csv(os.path.join(target_dir, f'{partition}.csv'), index=False)

# retain the copyright notice
shutil.copyfile(os.path.join(DATA_DIR, 'downloaded', 'alm', 'Potter', 'readme.txt'),
                os.path.join(DATA_DIR, 'tales_va', 'readme.txt'))
rmtree(tmp_dir)