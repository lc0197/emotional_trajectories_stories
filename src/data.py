
from datasets import load_dataset, disable_caching
import os
import pandas as pd
from tqdm import tqdm

from transformers import PreTrainedTokenizer

from config import DATA_DIR
import numpy as np

from collections import Counter


def create_sentence_token_strings(dataset, window_size, tokenizer:PreTrainedTokenizer):
    new_texts = {}
    labels = {}
    #centre_positions = []
    centre_positions = {}
    print('Creating strings for stories...')
    for csv in [os.path.join(DATA_DIR, dataset, f'{part}.csv') for part in ['train', 'val', 'test']]:
        df = pd.read_csv(csv)
        for col in ['V_EWE', 'A_EWE']:
            if not col in df.columns:
                df[col] = 0.5
                print(f'Warning: no gold standard columns (V_EWE, A_EWE) found in {csv}. This is fine if you only '
                      f'want to predict valence/arousal values. Training, however, will not work properly.')
        for story in tqdm(sorted(list(set(df.story.values)))):
            story_df = df[df.story == story]
            story_texts = list(story_df.text.values)
            text_ids = list(story_df.ID.values)
            all_labels = story_df[['V_EWE', 'A_EWE']].values
            for i, (text_id, t) in enumerate(zip(text_ids, story_texts)):
                text_ok = False
                # use this to make text shorter (less sentences), if necessary. I.e., take away sentences from the left side, right is truncated anyways
                reduce_right = 0
                reduce_left = 0
                new_text = ""
                # used to track the central positoin
                initial_left_boundary = max(i - window_size, 0)
                while not text_ok:

                    left_boundary = max(i-window_size+reduce_left, 0)
                    right_boundary = min((i+window_size+1) - reduce_right, len(story_texts))
                    sentences = [t.strip() for t in story_texts[left_boundary: right_boundary]]
                    ls = all_labels[left_boundary: right_boundary]
                    sep = " " + tokenizer.sep_token + " "
                    new_text = sep.join(sentences)
                    tokenized = tokenizer(new_text, truncation=True, max_length=512)
                    num_seps = Counter(tokenized.input_ids)[tokenizer.sep_token_id]
                    ls = ls[:num_seps]

                    centre_position =  i - left_boundary
                    if num_seps > centre_position:
                        break
                    else:
                        if reduce_left < reduce_right:
                            reduce_left += 1
                        else:
                            reduce_right += 1

                new_texts[text_id] = new_text
                labels[text_id] = ls
                centre_positions[text_id] = centre_position
    return new_texts, labels, centre_positions

def create_hf_tokens_dataset(args, tokenizer: PreTrainedTokenizer):
    disable_caching()
    full_strings, labels, all_centre_positions = create_sentence_token_strings(args.dataset, args.window_size, tokenizer)
    centre_positions = {}
    # save centre positions
    # dev_df = pd.read_csv(os.path.join(DATA_DIR, args.dataset, 'val.csv'))
    # centre_positions_val = [centre_positions[row['ID']] for _,row in dev_df.iterrows()]
    # test_df = pd.read_csv(os.path.join(DATA_DIR, args.dataset, 'test.csv'))
    # centre_positions_test = [centre_positions[row['ID']] for _, row in test_df.iterrows()]

    dss = load_dataset(path=os.path.join(DATA_DIR, args.dataset),
                       data_files={p: f'{p}.csv' for p in ['train', 'val', 'test']})
    train_ids = pd.read_csv(os.path.join(DATA_DIR, args.dataset, 'train.csv')).ID.values
    dev_ids = pd.read_csv(os.path.join(DATA_DIR, args.dataset, 'val.csv')).ID.values
    test_ids = pd.read_csv(os.path.join(DATA_DIR, args.dataset, 'test.csv')).ID.values
    id_map = {'train': train_ids, 'val': dev_ids, 'test':test_ids}

    for p, ds in dss.items():
        # tokenize
        dss[p] = dss[p].map(lambda x: tokenizer(full_strings[x['ID']], truncation=True, max_length=512), load_from_cache_file=False)
        # labels
        dss[p] = dss[p].map(lambda x: {'labels': create_token_labels(input_ids=x['input_ids'],
                                                          labels=labels[x['ID']],
                                                          sep_idx=tokenizer.sep_token_id,
                                                                     max_len =512)}, load_from_cache_file=False)
        centre_positions[p] = [all_centre_positions[i] for i in id_map[p]]
        # mask for tokens

    if args.debug_mode:
        generator = np.random.default_rng(args.seed)
        for p, ds in dss.items():
            all_idxs = list(range(len(ds)))
            selected = generator.choice(all_idxs, size=5, replace=False)
            dss[p] = ds.select(selected)
    label_cols = ['V_EWE', 'A_EWE']
    label2id = {l: i for i, l in enumerate(label_cols)}
    id2label = {i: l for i, l in enumerate(label_cols)}
    return dss['train'], dss['val'], dss['test'], label2id, id2label, centre_positions


def create_token_labels(input_ids, labels, sep_idx, max_len=512, dummy_value=-100.):
    dummy_label = np.full(shape=(2,), fill_value=dummy_value)
    ls = []
    label_ptr = 0
    for i in input_ids:
        if i==sep_idx:
            ls.append(labels[label_ptr])
            label_ptr += 1
        else:
            ls.append(dummy_label)
    labels = np.vstack(ls)
    #print(labels.shape[0])
    labels = np.pad(labels, ((0, max_len - labels.shape[0]), (0, 0)), constant_values=dummy_value)
    return labels
