import json
import os
from argparse import ArgumentParser
from shutil import rmtree
from time import time

import numpy as np
import pandas as pd
from scipy.special import expit
from transformers import AutoTokenizer, TrainingArguments, AutoModelForTokenClassification

from config import LOG_DIR, PREDICTION_DIR, DATA_DIR
from data import create_hf_tokens_dataset
from train import CustomTokenTrainer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Must point directly to a checkpoint directory, '
                                                                          'either under logs/ '
                                                                          '(typically something like logs/{dataset}/../checkpoint-1234) '
                                                                          'or on the hub.')
    parser.add_argument('--window_size', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    # TODO either --dataset or --input_csv?
    parser.add_argument('--dataset', type=str, help='Either --dataset or --input_csv must be given. '
                                                                         '--dataset is supposed to denote a directory below data/, '
                                                                         'containing train.csv, val.csv and test.csv.')
    parser.add_argument('--input_csv', type=str, help='Either --input_csv or --dataset must be given. The csv must have '
                                                      'columns "story" and "text", in accorance with the dataset format (cf. data/).'
                                                      'If --input_csv is given, --output_csv must be given, too.')
    parser.add_argument('--output_csv', type=str, help='Absolute path to an output csv, if --input_csv is given.')
    parser.add_argument('--debug_mode', action='store_true')
    args = parser.parse_args()

    # check for correct input format
    assert (args.dataset is None) ^ (args.input_csv is None), "Either --dataset or --input_csv must be given"
    if not (args.input_csv is None):
        assert not (args.output_csv is None), "If --input_csv is given, --output_csv must be, too."
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(args.output_csv)), 'hf_out')
    else:
        args.output_dir = os.path.join(PREDICTION_DIR, args.dataset, args.checkpoint_dir)
        os.makedirs(args.output_dir, exist_ok=True)

    # TODO hub: load window_size and base model elsewhere, if not hub
    checkpoint_dir = os.path.join(LOG_DIR, args.checkpoint_dir)
    if os.path.exists(checkpoint_dir):
        args.checkpoint_dir = checkpoint_dir
    else:
        print(f'Did not find {checkpoint_dir} locally, will try to load model {args.checkpoint_dir} from the hub.')

    #args.model_type = os.path.basename(Path(args.checkpoint_dir).parent.parent.parent.parent)
    # load config.json and add the parameters to args
    # TODO update this
    # config_file = os.path.join(Path(args.checkpoint_dir).parent.parent, 'config.json')
    # assert os.path.exists(config_file), f"{config_file} does not exist"
    # with open(config_file, 'r') as f:
    #     args_dct = json.load(f)
    # args_dct.update(vars(args))

    return args

# TODO simplify
def create_datasets(args, tokenizer):
    # TODO what if dev, test does not exist?
    train_ds, dev_ds, test_ds, _, _, central_sentences = create_hf_tokens_dataset(args, tokenizer)
    return train_ds, dev_ds, test_ds, central_sentences


# TODO simplify
def create_model(args, label2id, id2label):
    return AutoModelForTokenClassification.from_pretrained(args.checkpoint_dir, num_labels=len(label2id),
                                                        id2label=id2label, label2id=label2id,
                                                        ignore_mismatched_sizes=False)

# TODO simplify
def create_tokenizer(args):
    return AutoTokenizer.from_pretrained(args.checkpoint_dir)


def create_pseudo_dataset(input_csv):
    # validate input_csv
    required_cols = ['ID', 'story', 'text']
    df = pd.read_csv(input_csv)
    for c in required_cols:
        assert c in df.columns, f'Missing required column {c} in {input_csv}'
    # pseudo devel and test
    devel = df.iloc[:3,:]
    test = df.iloc[:3,:]
    pseudo_name = f'temp_{time()}'
    pseudo_dir = os.path.join(DATA_DIR, pseudo_name)
    os.makedirs(pseudo_dir)
    df.to_csv(os.path.join(pseudo_dir, 'train.csv'), index=False)
    devel.to_csv(os.path.join(pseudo_dir, 'val.csv'), index=False)
    test.to_csv(os.path.join(pseudo_dir, 'test.csv'), index=False)
    return pseudo_name


def create_trainer(model, args):
    training_arguments = TrainingArguments(per_device_train_batch_size=args.batch_size,
                                           per_device_eval_batch_size=args.batch_size,
                                           evaluation_strategy='epoch',
                                           output_dir=args.output_dir
                                           )
    return CustomTokenTrainer(model=model,
                train_dataset=train_ds,
                eval_dataset=dev_ds,
                tokenizer=tokenizer,
                compute_metrics=None,
                args=training_arguments)


def postprocess_prediction_tuple(pred_tuple, args, centre_positions):
    logits = pred_tuple.predictions # DS, max_SL, 2
    labels = pred_tuple.label_ids # DS, max_SL, 2

    # extract the [SEP] predictions from all token predictions
    sep_logits = []
    sep_labels = []
    for i in range(logits.shape[0]): # iterate over batch size
        logs = logits[i, :, :]  # max_SL, 2
        labs = labels[i, :, :] # max_SL, 2
        valid_idxs = np.argwhere(labs[:, 0] > -100.).flatten() # act_SL,
        valid_logs = logs[valid_idxs] # act_SL,
        sep_logits.append(valid_logs)
        valid_labels = labs[valid_idxs]
        sep_labels.append(valid_labels)
        # print(valid_idxs)

    #num_predictions = [l.shape[0] for l in sep_labels]


    # only consider the central sentences
    logits = [sep_logits[i][centre_positions[i], :] for i in range(len(centre_positions))]
    logits = np.vstack(logits)
    labels = [sep_labels[i][centre_positions[i], :] for i in range(len(centre_positions))]
    labels = np.vstack(labels)
    predictions = expit(logits)
    return (predictions[:,0], predictions[:,1]), (labels[:,0], labels[:,1])


if __name__ == '__main__':
    args = parse_args()

    label2id = {'V_EWE':0, 'A_EWE':1}
    id2label = {0:'V_EWE', 1:'A_EWE'}

    model = create_model(args, label2id, id2label)
    tokenizer = create_tokenizer(args)

    # data creation: pseudo dataset if --input_csv instead of --dataset
    if args.dataset is None:
        args.dataset = create_pseudo_dataset(args.input_csv)
    train_ds, dev_ds, test_ds, centre_positions = create_datasets(args, tokenizer)

    trainer = create_trainer(model, args)

    train_pred_tuple = trainer.predict(train_ds)
    (train_v_preds, train_a_preds), (train_v_labels, train_a_labels) = \
            postprocess_prediction_tuple(train_pred_tuple, args, centre_positions['train'])
    train_df = pd.DataFrame({'V_pred': train_v_preds,
                               'A_pred': train_a_preds
                               })

    dev_pred_tuple = trainer.predict(dev_ds)
    (dev_v_preds, dev_a_preds), (dev_v_labels, dev_a_labels) = postprocess_prediction_tuple(dev_pred_tuple, args, centre_positions['val'])
    #predictions = dev_pred_tuple.predictions

    dev_df = pd.DataFrame({'V_pred': dev_v_preds,
                       'A_pred': dev_a_preds
                           })

    test_pred_tuple = trainer.predict(test_ds)
    (test_v_preds, test_a_preds), (test_v_labels, test_a_labels) = postprocess_prediction_tuple(test_pred_tuple, args, centre_positions['test'])
    test_df = pd.DataFrame({'V_pred': test_v_preds,
                           'A_pred': test_a_preds})
    # save them in both prediction and log dir
    dfs_list = [train_df, dev_df, test_df] if args.input_csv is None else [train_df]
    part_names = ['train', 'val', 'test'] if args.input_csv is None else ['train']


    for df,name in zip(dfs_list, part_names):
        orig_df = pd.read_csv(os.path.join(DATA_DIR, args.dataset, f'{name}.csv'))
        if args.debug_mode:
            orig_df = orig_df.iloc[:20,:].copy()
        assert len(orig_df) == len(df)
        final_df = pd.concat([orig_df, df], axis=1)
        if args.input_csv is None:
            final_df.to_csv(os.path.join(args.output_dir, f'{args.dataset}_{name}_preds.csv'), index=False)
        else:
            final_df.to_csv(args.output_csv, index=False)
    print(' ')

    # clean up pseudo dataset, if necessary
    if not (args.input_csv is None):
        rmtree(os.path.join(DATA_DIR, args.dataset))
        if os.path.exists(args.output_dir):
            rmtree(args.output_dir)
