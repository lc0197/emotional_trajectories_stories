# Modeling Emotional Trajectories in Written Stories Utilizing Transformers and Weakly Supervised Learning
Repository for the paper *Modeling Emotional Trajectories in Written Stories Utilizing Transformers and Weakly Supervised Learning*, accepted into Findings of ACL 2024. 

ArXiv: [http://arxiv.org/abs/2406.02251](http://arxiv.org/abs/2406.02251)

BibTex:
```
TODO
```

## Installation
Please install the packages listed in ``requirements.txt``. 

In order to obtain the datasets, first run ``src/create_alm.py``. Upon completion, run ``src/create_gb.py``. 
Using custom datasets for training and prediction is possible, they only need to be in the same format as 
the provided ones. Place your own ``train.csv``, ``val.csv`` and ``test.csv`` in a new folder under ``data``. 
For the csv file format, please refer to the provided files.


## Training 
The training script is ``train.py``. Example: 

`` 
python3 train.py --dataset tales_va --seed 101 --num_seeds 5 --batch_size 4 --learning_rate 0.00003 --max_epochs 10 
--patience 2 --window_size 8
``
The ``--window_size`` argument refers to the number of sentences considered as context. 
Please refer to the ``parse_args`` method in ``train.py`` or call ``python3 train.py --help`` for an explanation of all parameters.


## Prediction 

Example for prediction script: 
``
python3 predict.py --dataset gutenberg_train --checkpoint_dir tales_va/trained_model/101/checkpoint-5061 --window_size 4 --batch_size 4
``

### Model checkpoints for prediction
The prediction script needs to be given either
* a previously saved checkpoint in the usual huggingface checkpoint format, saved in a a directory relative to ``logs/`` (for example call see above)
* a model as provided on the huggingface hub (see **Available checkpoints** below for their respective performance):
  - [chrlukas/stories-emotion-c0](https://huggingface.co/chrlukas/stories-emotion-c0) (model finetuned on Gb + Alm, context size ``0``)
  - [chrlukas/stories-emotion-c1](https://huggingface.co/chrlukas/stories-emotion-c1) (model finetuned on Gb + Alm, context size ``1``)
  - [chrlukas/stories-emotion-c2](https://huggingface.co/chrlukas/stories-emotion-c2) (model finetuned on Gb + Alm, context size ``2``)
  - [chrlukas/stories-emotion-c3](https://huggingface.co/chrlukas/stories-emotion-c4) (model finetuned on Gb + Alm, context size ``4``)
  - [chrlukas/stories-emotion-c4](https://huggingface.co/chrlukas/stories-emotion-c8) (model finetuned on Gb + Alm, context size ``8``, recommended)
    
  E.g. 
 ``
python3 predict.py --dataset gutenberg_train --checkpoint_dir chrlukas/stories-emotion-c4 --window_size 4 --batch_size 4
``
### Prediction Inputs
There are two options:
* predict for an existing dataset via the argument ``dataset`` (cf. example calls above). The resulting predictions are placed as ``.csv`` files in a directory ``predictions/{dataset}/{checkpoint dir}``.
* alternatively, a combination of ``--input_csv`` and ``--output_csv`` can be given, where ``input_csv`` has to have columns ``ID``, ``story`` and ``text``, analogous to the csvs making up a dataset in ``datasets``.
Exemplary input csv:

| ID  | story  | text                                         |
|-----|--------|----------------------------------------------|
| 1   | story1 | This is the first sentence of test story 1.  |
| 2   | story1 | This is the second sentence of test story 1. |
| ... | ...    | ...                                          |
| 42  | story1 | This is the final sentence of test story 1.  |
| 43  | story2 | Here, another test story starts.             |
| ... | ...    | ...                                          |

Example call:

``
python3 predict.py --input_csv input_file.csv --output_csv output_file.csv --checkpoint_dir chrlukas/stories-emotion-c4 --window_size 4 --batch_size 4
``
The output csv will be a copy of the input value, extended by two additional columns ``V_pred`` and ``A_pred`` for the predicted valence/arousal values. Example output: 

Note that the two options are incompatible, i.e. either ``--dataset`` or ``--input_csv`` and ``--output_csv`` must be given.


## Available Checkpoints 
Trained models are available on the huggingface hub: 

| Model                                                                  | Valence dev/test   | Arousal dev/test   |
|------------------------------------------------------------------------|--------------------|--------------------|
|[stories-emotion-c0](https://huggingface.co/chrlukas/stories-emotion-c0)| .7091/.7187        | .5815/.6189        |
|[stories-emotion-c1](https://huggingface.co/chrlukas/stories-emotion-c1)| .7715/.7875        | .6458/.6935        |
|[stories-emotion-c2](https://huggingface.co/chrlukas/stories-emotion-c2)| .7922/.8074        | .6667/.6954        |
|[stories-emotion-c4](https://huggingface.co/chrlukas/stories-emotion-c4)| .8078/.8146        | .6763/.7115        |
|[stories-emotion-c8](https://huggingface.co/chrlukas/stories-emotion-c8)| **.8223**/**.8237**| **.6829**/**.7120**|

We provide the best out of 5 seeds for each context size. Hence, the numbers in this table differ from the result table in the paper, where the mean performance across 5 seeds is reported.
