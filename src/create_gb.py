import os
import subprocess
from shutil import copyfile

import pandas as pd
import re
from tqdm import tqdm
from glob import glob
import pysbd

from config import RES_DIR, DATA_DIR


def download_ebook(ebook_id, target_dir):
    link = f'https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt'
    ret = subprocess.run(f'wget {link} --output-document={os.path.join(target_dir, str(ebook_id))}.txt', shell=True)
    return ret.returncode

def cut_ebook(e_id, source_dir, target_dir):
    edf = cut_df[cut_df.ebook_id==e_id]
    stories = sorted(list(set(edf.story_name.values)))
    with open(os.path.join(source_dir, f'{e_id}.txt'), 'r') as f:
        lines = [l.replace("\n", " ") for l in f.readlines()]
    for story in stories:
        story_lines = []
        story_df = edf[edf.story_name == story]
        for _,row in story_df.iterrows():
            story_lines.extend(lines[row.line_start_in - 1: row.line_end_ex - 1])
        target_file = os.path.join(target_dir, f'{e_id}_{story}.txt')
        with open(target_file, 'w+') as f:
            f.writelines(story_lines)


# auxiliary method to handle quotation marks in certain strings
def fix_quotation(s, seg):
    new_s = []
    prefix_ptr = 0
    search_start_ptr = 0
    match = re.search(quotation_re, s[search_start_ptr:])
    if match is None:
        return [s]
    while match != None:
        span = match.span()
        # print(f'Found match: {match}')
        prefix = s[prefix_ptr: search_start_ptr + span[0]]
        # print(f'Prefix: {prefix}')
        quotation = s[search_start_ptr + span[0]:search_start_ptr + span[1]]
        # print(f'Quotation: {quotation}')
        suffix = s[search_start_ptr + span[1]:]
        # print(f'Suffix: {suffix}')
        # split quotation (without quotation marks)
        split_q = seg.segment(quotation[1:-1])
        # print(f'Quotation split: {split_q}')
        # actual split happened
        # print(f'Split of length {len(split_q)}')
        if len(split_q) > 1:

            new_s.append(prefix + '"' + split_q[0])
            # print(f'Appended {new_s[-1]} to set of sentences')
            if len(split_q) >= 3:
                new_s.extend(split_q[1:-1])
                # print(f'Extended set of sentences by {split_q[1:-1]}')
            new_s.append(split_q[-1] + suffix)
            # print(f'Appended {new_s[-1]} to set of sentences')
            # done
            prefix_ptr = len(s)

        search_start_ptr = search_start_ptr + span[1]
        # print(f'Continuing search on {s[search_start_ptr:]}')
        # print(f'Prefix pointer: {prefix_ptr}')
        match = re.search(quotation_re, s[search_start_ptr:])
        # print()

    if prefix_ptr < len(s):
        # print(f'Done, appending remainder: {s[prefix_ptr:]}')
        new_s.append(s[prefix_ptr:])
    return [s.strip() for s in new_s]


if __name__ == '__main__':
    assert os.path.exists(os.path.join(DATA_DIR, 'tales_va', 'val.csv')), "Please run create_alm.py first"
    assert os.path.exists(os.path.join(DATA_DIR, 'tales_va', 'test.csv')), "Please run create_alm.py first"

    raw_target_dir = os.path.join(DATA_DIR, 'downloaded', 'gutenberg', 'gutenberg_raw')
    os.makedirs(raw_target_dir, exist_ok=True)

    with open(os.path.join(RES_DIR, 'ebook_ids.txt'), 'r') as f:
        ebook_ids = [l.replace("\n", "") for l in f.readlines()]

    # download
    print('Downloading ebooks...')
    for ebook_id in tqdm(ebook_ids):
        r = download_ebook(ebook_id, target_dir=raw_target_dir)
        print(ebook_id, r)

    # cut into stories
    print('Cutting books into stories')
    cut_target_dir = os.path.join(DATA_DIR, 'downloaded', 'gutenberg', 'ebooks')
    os.makedirs(cut_target_dir, exist_ok=True)
    cut_df = pd.read_csv(os.path.join(RES_DIR, 'ebook_cuts.csv'))
    for ebook_id in tqdm(ebook_ids):
        cut_ebook(int(ebook_id), source_dir=raw_target_dir, target_dir=cut_target_dir)

    # clean up
    bracket_re = re.compile(r'\[.+?\]')
    empty_re = re.compile('\s\s+')
    quotation_re = r'\"[^\"]+\"'

    seg = pysbd.Segmenter(language='en', clean=False)
    txts = sorted(glob(os.path.join(cut_target_dir, '*.txt')))

    target_dir = os.path.join(DATA_DIR, 'gutenberg_train')
    os.makedirs(target_dir, exist_ok=True)

    # clean and build train.csv from the gutenberg data
    story_dfs = []
    id_ptr = 100000
    print('Splitting stories into sentences...')
    for txt in tqdm(txts):
        with open(txt, 'r') as f:
            text = f.readlines()
        story_name = os.path.basename(txt).replace(".txt","")
        assert len(text) == 1
        text = text[0]
        text = re.sub(bracket_re, "", text)
        text = text.replace("--", " - ")
        text = re.sub(empty_re, " ", text)
        text = text.replace("_", "")
        for a in ['“', '”']:
            text = text.replace(a, '"')
        text = text.replace('" "', '"\n"')
        sentences = seg.segment(text)
        # quotations
        new_sentences = []
        for s in sentences:
            new_sentences.extend(fix_quotation(s, seg=seg))
        story_dfs.append(pd.DataFrame({
            'ID': list(range(id_ptr, id_ptr+len(new_sentences))),
            'story': [story_name] * len(new_sentences),
            'text': new_sentences
        }))
        id_ptr = id_ptr + len(new_sentences)
    train_df = pd.concat(story_dfs)
    train_df.to_csv(os.path.join(target_dir, 'train.csv'), index=False)

    # copy reduced versions of dev and test from tales_va
    for partition in ['val', 'test']:
        tgt_file = os.path.join(target_dir, f'{partition}.csv')
        copyfile(os.path.join(DATA_DIR, 'tales_va', f'{partition}.csv'), tgt_file)
        part_df = pd.read_csv(tgt_file)[['ID', 'story', 'text', 'V_EWE', 'A_EWE']]
        part_df.to_csv(tgt_file, index=False)