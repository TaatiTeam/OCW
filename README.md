# Only Connect Wall (OCW) Dataset

The accompanying repo for the Only Connect Wall (OCW) Dataset.

TODO: link to the paper

## Usage

### Downloading the dataset

The dataset can be downloaded from [here](https://www.cs.toronto.edu/~taati/OCW/OCW.tar.gz) or with bash script:
    
```bash
bash download_OCW.sh
```

There are four files/partitions:  `OCW.json`, `train.json`, `validation.json` and `test.json`.

Here is an example of the dataset's structure:


```json
{
	"season_to_walls_map": {
		"0": 0,
		"1": 30,
		"2": 16,
		"3": 30,
		"4": 32,
		"5": 32,
		"6": 32,
		"7": 26,
		"8": 26,
		"9": 26,
		"10": 54,
		"11": 54,
		"12": 74,
		"13": 74,
		"14": 56,
		"15": 56
	},
	"dataset": [{
		"wall_id": "882c",
		"season": 1,
		"episode": 5,
		"words": ["Puzzle", "Manhattan", "B", "Wrench", "Smith", "Nuts", "Brooks", "Blanc", "Suit", "Screwdriver", "Sidecar", "Margarita", "Hammer", "Business", "Gimlet", "Gibson"],
		"gt_connections": ["Famous Mels", "Household tools", "Cocktails", "Monkey ___"],
		"groups": {
			"group_1": {
				"group_id": "882c_01",
				"gt_words": ["Blanc", "Brooks", "B", "Smith"],
				"gt_connection": "Famous Mels",
				"human_performance": {
					"grouping": 1,
					"connection": 1
				}
			},
			"group_2": {
				"group_id": "882c_02",
				"gt_words": ["Screwdriver", "Hammer", "Gimlet", "Wrench"],
				"gt_connection": "Household tools",
				"human_performance": {
					"grouping": 1,
					"connection": 1
				}
			},
			"group_3": {
				"group_id": "882c_03",
				"gt_words": ["Sidecar", "Manhattan", "Gibson", "Margarita"],
				"gt_connection": "Cocktails",
				"human_performance": {
					"grouping": 1,
					"connection": 1
				}
			},
			"group_4": {
				"group_id": "882c_04",
				"gt_words": ["Puzzle", "Business", "Nuts", "Suit"],
				"gt_connection": "Monkey ___",
				"human_performance": {
					"grouping": 1,
					"connection": 1
				}
			}
		},
		"overall_human_performance": {
			"grouping": [1, 1, 1, 1],
			"connections": [1, 1, 1, 1]
		}
	}]
}
```

`"season_to_walls_map"` contains counts for the number of walls in each season. `"dataset"` is a list of dictionaries, where each dictionary contains all accompanying information about a wall, including the `"groups"` and each groups ground truth words (`"gt_words"`) and ground truth connections (`"gt_connections"`). Each wall and group has a unique ID. Other metadata, such as human performance, is also included.

### Loading the dataset

The three partitions can be loaded the same way as any other JSON file. For example, using Python:

```python
dataset = {
    "train": json.load(open("train.json", "r"))["dataset"],
    "validation": json.load(open("validation.json", "r"))["dataset"],
    "test": json.load(open("test.json", "r"))["dataset"],
}
```

However, it is likely easiest to work with the dataset using the HuggingFace Datasets library:

```python
# pip install datasets
dataset = load_dataset(
    "json",
    data_files={
        "train": "train.json",
        "validation": "validation.json",
        "test": "test.json",
    },
    field="dataset",
)
```

### Evaluating

We provide a script for evaluating the performance of a model on the dataset. Before running, make sure you have installed the requirements:

```bash
pip install -r requirements.txt
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
python evaluate_only_connect.py \
    --prediction_file "./predictions/task1.json" \
    --dataset_path "./dataset/" \
    --results_path "./results/" \
    --task "task1-grouping"
```

### Running the baselines

#### Word Embeddings and Pre-trained Language Models

To run word embeddings and PLM baseline:
    
```bash
python prediction.py \
    --model_name "intfloat/e5-base-v2" \
    --dataset_path "./dataset/" \
    --predictions_path "./predictions/" \
    --task "task1-grouping"
```
The `model_name` should be from huggingface model hub or in `['elmo', 'glove', 'crawl', 'news']`.
To run contextualized embeddings in PLMs, use `--contextual` flag.

To plot the results:

```bash
python plot.py \
    --wall_id "8cde" \
    --model_name "intfloat/e5-base-v2" \
    --shuffle_seed 9
```

#### Large Language Models

To run the few-shot in-context LLM baseline, see the [`run_openai.ipynb`](run_openai.ipynb) notebook. Note: this will require an OpenAI API key.

## Contributing

We welcome contributions to this repository (noticed a typo? a bug?). To propose a change:

```
git clone https://github.com/salavina/OCW
cd OCW
git checkout -b my-branch
pip install -r requirements.txt
pip install -r dev-requirements.txt
```

Once your changes are made, make sure to lint and format the code (addressing any warnings or errors):

```
isort .
black .
flake8 .
```

Then, submit your change as a pull request.

## Citing

If you use the Only Connect dataset in your work, please consider citing our paper:

TODO

