# 🧩 Only Connect Wall (OCW) Dataset

The Only Connect Wall (OCW) dataset contains 618 _"Connecting Walls"_ from the [Round 3: Connecting Wall](https://en.wikipedia.org/wiki/Only_Connect#Round_3:_Connecting_Wall) segment of the [Only Connect quiz show](https://en.wikipedia.org/wiki/Only_Connect), collected from 15 seasons' worth of episodes. Each wall contains the ground-truth __groups__ and __connections__ as well as recorded human performance. Please see [our paper](https://arxiv.org/abs/2306.11167) for more details about the dataset and its motivations.

## 📋 Table of Contents

- [🧩 Only Connect Wall (OCW) Dataset](#-only-connect-wall-ocw-dataset)
  - [📋 Table of Contents](#-table-of-contents)
  - [📖 Usage](#-usage)
    - [Downloading the dataset](#downloading-the-dataset)
    - [Dataset structure](#dataset-structure)
    - [Loading the dataset](#loading-the-dataset)
    - [Evaluating](#evaluating)
    - [Downloading easy datasets for ablation studies](#downloading-easy-datasets-for-ablation-studies)
    - [Running the baselines](#running-the-baselines)
      - [Word Embeddings and Pre-trained Language Models](#word-embeddings-and-pre-trained-language-models)
      - [Large Language Models](#large-language-models)
  - [✍️ Contributing](#️-contributing)
  - [📝 Citing](#-citing)
  - [🙏 Acknowledgements](#-acknowledgements)

## 📖 Usage

### Downloading the dataset

The dataset can be downloaded from [here](https://www.cs.toronto.edu/~taati/OCW/OCW.tar.gz) or with a bash script:
    
```bash
bash download_OCW.sh
```

### Dataset structure

The dataset is provided as JSON files, one for each partition: `train.json`, `validation.json` and `test.json`. We also provide a `OCW.json` file that contains all examples across all splits. The splits are sized as follows:

| Split | # Walls | 
|:-------|:---------:|
| `train` |   62   |
| `validation`   | 62      |
| `test`  | 494      |

Here is an example of the dataset's structure:

```json
{
  "season_to_walls_map": {
    "1": {
      "num_walls": 30,
      "start_date": "15/09/2008",
      "end_date": "22/12/2008"
    }
  },
  "dataset": [
    {
      "wall_id": "882c",
      "season": 1,
      "episode": 5,
      "words": [
        "Puzzle",
        "Manhattan",
        "B",
        "Wrench",
        "Smith",
        "Nuts",
        "Brooks",
        "Blanc",
        "Suit",
        "Screwdriver",
        "Sidecar",
        "Margarita",
        "Hammer",
        "Business",
        "Gimlet",
        "Gibson"
      ],
      "gt_connections": [
        "Famous Mels",
        "Household tools",
        "Cocktails",
        "Monkey ___"
      ],
      "groups": {
        "group_1": {
          "group_id": "882c_01",
          "gt_words": [
            "Blanc",
            "Brooks",
            "B",
            "Smith"
          ],
          "gt_connection": "Famous Mels",
          "human_performance": {
            "grouping": 1,
            "connection": 1
          }
        },
        "group_2": {
          "group_id": "882c_02",
          "gt_words": [
            "Screwdriver",
            "Hammer",
            "Gimlet",
            "Wrench"
          ],
          "gt_connection": "Household tools",
          "human_performance": {
            "grouping": 1,
            "connection": 1
          }
        },
        "group_3": {
          "group_id": "882c_03",
          "gt_words": [
            "Sidecar",
            "Manhattan",
            "Gibson",
            "Margarita"
          ],
          "gt_connection": "Cocktails",
          "human_performance": {
            "grouping": 1,
            "connection": 1
          }
        },
        "group_4": {
          "group_id": "882c_04",
          "gt_words": [
            "Puzzle",
            "Business",
            "Nuts",
            "Suit"
          ],
          "gt_connection": "Monkey ___",
          "human_performance": {
            "grouping": 1,
            "connection": 1
          }
        }
      },
      "overall_human_performance": {
        "grouping": [
          1,
          1,
          1,
          1
        ],
        "connections": [
          1,
          1,
          1,
          1
        ]
      }
    }
  ]
}
```

where

- `"season_to_walls_map"` contains the `"num_walls"` in each season, as well as the `"start_date"` and `"end_date"` the season ran
- `"dataset"` is a list of dictionaries, where each dictionary contains all accompanying information about a wall:
  - `"wall_id"`: a unique string identifier for the wall
  - `"season"`: an integer representing the season the wall was collected from
  - `"episode"`: an integer representing the episode the wall was collected from
  - `"words"`: a list of strings representing the words in the wall in random order
  - `"gt_connections"`: a list of strings representing the ground truth connections of each group
  - `"groups`: a dictionary of dictionaries containing the four groups in the wall, each has the following items:
      - `"group_id"`: a unique string identifier for the group
      - `"gt_words"`: a list of strings representing the ground truth words in the group
      - `"gt_connection"`: a string representing the ground truth connection of the group
      - `"human_performance`: a dictionary containing recorded human performance for the grouping and connections tasks
  - `"overall_human_performance"`: a dictionary containing recorded human performance for the grouping and connections tasks for each group in the wall

### Loading the dataset

The three partitions can be loaded the same way as any other JSON file. For example, using Python:

```python
dataset = {
    "train": json.load(open("./dataset/train.json", "r"))["dataset"],
    "validation": json.load(open("./dataset/validation.json", "r"))["dataset"],
    "test": json.load(open("./dataset/test.json", "r"))["dataset"],
}
```

However, it is likely easiest to work with the dataset using the [HuggingFace Datasets](https://huggingface.co/datasets) library:

```python
# pip install datasets
from datasets import load_dataset

dataset = load_dataset("TaatiTeam/OCW")

# The dataset can be used like any other HuggingFace dataset
# E.g. get the wall_id of the first example in the train set
dataset["train"]["wall_id"][0]
# or get the words of the first 10 examples in the test set
dataset["test"]["words"][0:10]
```

> __Note__
> The HuggingFace copy does not include the `"season_to_walls_map"` 

### Evaluating

We provide a script for evaluating the performance of a model on the dataset. Before running, make sure you have installed the requirements and package:

```bash
pip install -r requirements.txt
pip install -e .
```

Then, ensure your model predictions are formatted as follows in a JSON file:

```json
[{
    "wall_id": "882c_01",
    "predicted_groups": [
        ["Puzzle", "Manhattan", "B", "Wrench"],
        ["Smith", "Nuts", "Brooks", "Blanc"],
        ["Suit", "Screwdriver", "Sidecar", "Margarita"],
        ["Hammer", "Business", "Gimlet", "Gibson"]
    ],
    "predicted_connections": ["Famous Mels", "Household tools", "Cocktails", "Monkey ___"]
}]
```

Note, only one of `"predicted_groups"` or `"predicted_connections"` is required. The other can be `null`. In the evaluate script, predicting groups is considered `"task1-grouping"` and predicting connections is considered `"task2-connections"`.

To run the evaluation script:

```bash
python src/ocw/evaluate_only_connect.py \
    --prediction-file "./predictions/task1.json" \
    --dataset-path "./dataset/" \
    --results-path "./results/" \
    --task "task1-grouping"
```

### Downloading easy datasets for ablation studies

We also produced two "easy" versions of the dataset, designed to remove or dramatically reduce the number of red herrings, for abalation:

- A copy of the dataset where each wall in the test set is replaced with a _random_ selection of groups. No group is repeated twice, and no wall contains two copies of the same clue. The train and validation sets are unmodified. This dataset can be downloaded from [here](https://www.cs.toronto.edu/~taati/OCW/OCW_randomized.tar.gz) or with a bash script:
    
```bash
bash download_OCW_randomized.sh
```

- A copy of the dataset generated from WordNet by selecting equivalent synonyms for each clue in a group. This dataset can be downloaded from [here](https://www.cs.toronto.edu/~taati/OCW/OCW_wordnet.tar.gz) or with a bash script:
    
```bash
bash download_OCW_wordnet.sh
```

Both of these datasets can be loaded from HuggingFace Datasets:

```python
# pip install datasets
from datasets import load_dataset

ocw_randomized = load_dataset("TaatiTeam/OCW", "ocw_randomized")
ocw_wordnet = load_dataset("TaatiTeam/OCW", "ocw_wordnet")
```

### Running the baselines

#### Word Embeddings and Pre-trained Language Models

To run word embeddings and PLM baseline:
    
```bash
python scripts/prediction.py \
    --model-name "intfloat/e5-base-v2" \
    --dataset-path "./dataset/" \
    --predictions-path "./predictions/" \
    --task "task1-grouping"
```
The `model_name` should be from huggingface model hub or in `['elmo', 'glove', 'crawl', 'news']`.
To run contextualized embeddings in PLMs, use `--contextual` flag.

To plot the results:

```bash
python scripts/plot.py \
    --wall-id "8cde" \
    --model-name "intfloat/e5-base-v2" \
    --shuffle-seed 9
```

#### Large Language Models

To run the few-shot in-context LLM baseline, see the [`run_openai.ipynb`](./notebooks/run_openai.ipynb) notebook. Note: this will require an OpenAI API key.

## ✍️ Contributing

We welcome contributions to this repository (noticed a typo? a bug?). To propose a change:

```
git clone https://github.com/salavina/OCW
cd OCW
git checkout -b my-branch
pip install -r requirements.txt
pip install -e .
```

Once your changes are made, make sure to lint and format the code (addressing any warnings or errors):

```
isort .
black .
flake8 .
```

Then, submit your change as a pull request.

## 📝 Citing

If you use the Only Connect dataset in your work, please consider citing our paper:

```
@article{Naeini2023LargeLM,
    title        = {Large Language Models are Fixated by Red Herrings: Exploring Creative Problem Solving and Einstellung Effect using the Only Connect Wall Dataset},
    author       = {Saeid Alavi Naeini and Raeid Saqur and Mozhgan Saeidi and John Giorgi and Babak Taati},
    year         = 2023,
    journal      = {ArXiv},
    volume       = {abs/2306.11167},
    url          = {https://api.semanticscholar.org/CorpusID:259203717}
}
```

## 🙏 Acknowledgements

We would like the thank the maintainers and contributors of the fan-made and run website [https://ocdb.cc/](https://ocdb.cc/) for providing the data for this dataset. We would also like to thank the creators of the Only Connect quiz show for producing such an entertaining and thought-provoking show.