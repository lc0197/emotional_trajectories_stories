import os
from argparse import ArgumentParser

import datasets
import torch
from audmetric import concordance_cc
from scipy.special import expit
from torch import nn
from transformers import AutoTokenizer, TrainingArguments, Trainer, \
    EarlyStoppingCallback, AutoModelForTokenClassification
import pandas as pd
import numpy as np
import json

import gc

from config import LOG_DIR, device
from data import create_hf_tokens_dataset


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--dataset', default='tales_vad', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_epochs', type=int, default=7)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--debug_mode', action='store_true', help='Use just a few data points')
    parser.add_argument('--run_name', type=str, required=False)
    parser.add_argument('--experiment_family', type=str, required=False)
    parser.add_argument('--checkpoint', type=str, default='microsoft/deberta-v3-large', required=True)
    parser.add_argument('--window_size', type=int, default=8)

    args = parser.parse_args()

    return args

# TODO get rid of this?
def create_tokenizer(args):
    return AutoTokenizer.from_pretrained(args.checkpoint)

# TODO get rid of this?
def create_datasets(args, tokenizer):
    return create_hf_tokens_dataset(args, tokenizer)


# TODO get rid of this
def create_model(args, label2id, id2label):
        model = AutoModelForTokenClassification.from_pretrained(args.checkpoint, num_labels = len(label2id),
                                                               id2label=id2label, label2id=label2id,
                                                                    ignore_mismatched_sizes=True)
        return model


def compute_cccs_for_tokens(eval_pred, central_sentences):
    logits, labels = eval_pred

    # extract the [SEP] predictions from all token predictions
    sep_logits = []
    sep_labels = []
    for i in range(logits.shape[0]):
        logs = logits[i,:,:] # SL, 2
        labs = labels[i,:,:]
        valid_idxs = np.argwhere(labs[:,0] > -100.).flatten()
        valid_logs = logs[valid_idxs]
        sep_logits.append(valid_logs)
        valid_labels = labs[valid_idxs]
        sep_labels.append(valid_labels)

    # which sentence is the central one?
    num_predictions = [l.shape[0] for l in sep_labels]
    print(f'num_predictions: {num_predictions}')

    # only consider the central sentences
    logits = [sep_logits[i][central_sentences[i],:] for i in range(len(central_sentences))]
    logits = np.vstack(logits)
    labels = [sep_labels[i][central_sentences[i],:] for i in range(len(central_sentences))]
    labels = np.vstack(labels)
    predictions = expit(logits)

    cccs = {f'{label}_CCC': concordance_cc(labels[:,i], predictions[:,i]) for i,label in id2label.items()}
    mean_ccc = np.mean(list(cccs.values()))
    cccs.update({'average_CCC': mean_ccc})
    return cccs


class CustomTokenTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        # remove labels from input, as their size (max_len) differs from that of the input IDs (max_batch_len)
        labels = inputs.pop("labels")# BS, max_len, 2
        model_out = model(**inputs)
        logits = model_out.logits # BS, SL, 2
        seq_len = logits.shape[1]
        labels = labels[:,:seq_len,:] # BS, SL, 2
        # determine relevant positions
        labels = labels.reshape(labels.shape[0]*labels.shape[1], labels.shape[2]) # BS*SL, 2
        label_idxs = torch.argwhere(labels[:,0] > -100.).squeeze() # BS*max_len,
        logits = logits.reshape(logits.shape[0]*logits.shape[1], logits.shape[2]) # BS*SL, 2
        # select only relevant labels and corresponding logits
        labels = torch.index_select(labels, dim=0, index=label_idxs) # num_labels_overall, 2
        logits = torch.index_select(logits, dim=0, index=label_idxs) # num_labels_overall, 2
        # compute loss
        loss_fn = nn.MSELoss()
        sigmoid = nn.Sigmoid()
        activated = sigmoid(logits)
        loss = loss_fn(activated, labels)
        return (loss, model_out) if return_outputs else loss


if __name__ == '__main__':
    args = parse_args()
    conf_dict = vars(args)

    tokenizer = create_tokenizer(args)
    train_ds, dev_ds, test_ds, label2id, id2label, centre_positions = create_datasets(args, tokenizer)

    experiment_dir = os.path.join(LOG_DIR, args.dataset, args.run_name)
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.json'), 'w+') as f:
        json.dump(conf_dict, f)

    # TODO remove?
    collator = None

    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)]

    log_dct = {}

    for seed in range(args.seed, args.seed+args.num_seeds):
        torch.random.manual_seed(args.seed)
        model = create_model(args, label2id=label2id, id2label=id2label)

        model.to(device)

        seed_dir = os.path.join(experiment_dir, str(seed))
        training_arguments = TrainingArguments(per_device_train_batch_size=args.batch_size,
                                           per_device_eval_batch_size=2*args.batch_size,
                                           evaluation_strategy='epoch',
                                           learning_rate=args.learning_rate,
                                           num_train_epochs=args.max_epochs,
                                           logging_strategy='epoch',
                                           save_strategy='epoch',
                                               seed=seed,
                                               data_seed=seed,
                                               load_best_model_at_end=True,
                                               metric_for_best_model='average_CCC',
                                               greater_is_better=True,
                                               output_dir= seed_dir,
                                               report_to="tensorboard",
                                               save_total_limit=1
                                           )
        trainer_class = CustomTokenTrainer
        trainer = trainer_class(model=model,
                          data_collator=collator,
                          train_dataset=train_ds,
                          eval_dataset=dev_ds,
                          tokenizer=tokenizer,
                          compute_metrics=lambda x: compute_cccs_for_tokens(x, central_sentences=centre_positions['val']),
                          callbacks=callbacks,
                          args=training_arguments)

        trainer.train()

        dev_results = trainer.predict(dev_ds, metric_key_prefix='dev')
        dev_metrics = dev_results.metrics
        dev_preds = dev_results.predictions

        # necessary to consider the centre positions for test
        trainer.compute_metrics = lambda x: compute_cccs_for_tokens(x, central_sentences=centre_positions['test'])
        test_results = trainer.predict(test_ds, metric_key_prefix='test')
        test_metrics = test_results.metrics
        test_preds = test_results.predictions

        seed_dct = dev_metrics
        seed_dct.update(test_metrics)
        log_dct[seed] = seed_dct

        del (model)
        gc.collect()

    # summarise results
    summ_dct = {}
    for k in log_dct[args.seed].keys():
        values = [log_dct[i][k] for i in log_dct.keys()]
        for m,f in zip(['mean', 'std'], [np.mean, np.std]):
            summ_dct[f'{k}_{m}'] = f(values)
    # save summary as json
    with open(os.path.join(experiment_dir, 'summary.json'), 'w+') as f:
        json.dump(summ_dct, f)
    # add to experiment family
    if not (args.experiment_family is None):
        row_dct = conf_dict
        row_dct.update(summ_dct)
        row_df = pd.DataFrame({k:[v] for k,v in row_dct.items()})
        family_csv = os.path.join(LOG_DIR, args.dataset, f'{args.experiment_family}.csv')
        if os.path.exists(family_csv):
            df = pd.read_csv(family_csv)
            df = pd.concat([df, row_df])
        else:
            df = row_df
        df.to_csv(family_csv, index=False)

    print()



