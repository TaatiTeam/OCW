import json
import os

import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi
from scipy.stats import wasserstein_distance as wd
from tqdm.auto import tqdm

import utils
from arguments import get_args
from evaluate import load


class Evaluate:
    def __init__(self, prediction_file, dataset_path="./", results_path="./results", split="test", seed=42):
        self.prediction_json = prediction_file
        self.dataset_path = dataset_path
        self.results_path = results_path
        self.split = split
        self.seed = seed

        # Task 1 metrics
        self.WD = []
        self.NMI = []
        self.FULL_WALL = 0
        self.CORRECT_GROUPS = 0
        # Task 2 metrics
        self.EXACT_MATCH = []
        self.ROUGE = []
        self.BERT_SCORE = []

    def task1_evaluation(self):
        utils.set_seed(seed=self.seed)
        oc_eval_results = {"global": {}, "granular": []}
        if not os.path.exists(self.dataset_path):
            raise Exception("Dataset path does not exist")
        dataset = utils.load_hf_dataset(self.dataset_path)
        prediction = utils.load_prediction(self.prediction_json)
        for wall in tqdm(dataset[self.split]):
            gt_words = [i["gt_words"] for i in wall["groups"].values()]
            pred_words = utils.find_wall(wall["wall_id"], prediction)["predicted_groups"]
            gt_sorted = [sorted(i) for i in gt_words]
            pred_sorted = [sorted(i) for i in pred_words]
            correct_groups = utils.check_equal(pred_sorted, gt_sorted)
            self.CORRECT_GROUPS += correct_groups
            if correct_groups == 4:
                self.FULL_WALL += 1
            gt_lst = [item for sublist in gt_sorted for item in sublist]
            pred_lst = [item for sublist in pred_sorted for item in sublist]
            index_gt = utils.clue2group(gt_lst, gt_lst)
            index_pred = utils.clue2group(pred_lst, gt_lst)
            wd_val = wd(index_pred, index_gt)
            nmi_val = nmi(index_pred, index_gt)
            self.WD.append(wd_val)
            self.NMI.append(nmi_val)
            oc_eval_results["granular"].append(
                {
                    "wall_id": wall["wall_id"],
                    "WD": wd_val,
                    "NMI": nmi_val,
                    "correct_groups": correct_groups,
                    "full_wall": correct_groups == 4,
                }
            )
        oc_eval_results["global"]["WD"] = np.mean(self.WD)
        oc_eval_results["global"]["NMI"] = np.mean(self.NMI)
        oc_eval_results["global"]["full_wall"] = self.FULL_WALL
        oc_eval_results["global"]["correct_groups"] = self.CORRECT_GROUPS

        # save results as json
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
        with open(
            self.results_path + "/" + self.prediction_json.split("_")[0].split("/")[-1] + "_results.json", "w"
        ) as f:
            json.dump(oc_eval_results, f)

        print("results saved to: ", self.results_path)

    def task2_evaluation(self):
        utils.set_seed(seed=self.seed)
        exact_match = load("exact_match")
        rouge = load("rouge")
        bertscore = load("bertscore")
        oc_eval_results = {"global": {}, "granular": []}
        if not os.path.exists(self.dataset_path):
            raise Exception("Dataset path does not exist")
        dataset = utils.load_hf_dataset(self.dataset_path)
        prediction = utils.load_prediction(self.prediction_json)
        for wall in tqdm(dataset[self.split]):
            gt_connections = wall["gt_connections"]
            pred_connections = utils.find_wall(wall["wall_id"], prediction)["predicted_connections"]
            # Lowercase and strip so results are invariant to trailing whitespace and capitalization
            gt_connections = [connection.lower().strip() for connection in gt_connections]
            pred_connections = [connection.lower().strip() for connection in pred_connections]
            exact_match_results = [
                exact_match.compute(predictions=[pred], references=[gt])["exact_match"]
                for pred, gt in zip(pred_connections, gt_connections)
            ]
            rouge_results = [
                rouge.compute(predictions=[pred], references=[gt])["rouge1"]
                for pred, gt in zip(pred_connections, gt_connections)
            ]
            bert_score_results = bertscore.compute(
                predictions=pred_connections,
                references=gt_connections,
                lang="en",
                rescale_with_baseline=True,
                use_fast_tokenizer=True,
            )
            self.EXACT_MATCH.append(np.mean(exact_match_results))
            self.ROUGE.append(np.mean(rouge_results))
            self.BERT_SCORE.append(np.mean(bert_score_results["f1"]))
            oc_eval_results["granular"].append(
                {
                    "wall_id": wall["wall_id"],
                    "exact_match": exact_match_results,
                    "rouge1_f1": rouge_results,
                    "bert_score_f1": bert_score_results["f1"],
                    "bert_score_hashcode": bert_score_results["hashcode"],
                }
            )
        oc_eval_results["global"]["exact_match"] = np.mean(self.EXACT_MATCH)
        oc_eval_results["global"]["rouge1_f1"] = np.mean(self.ROUGE)
        oc_eval_results["global"]["bert_score_f1"] = np.mean(self.BERT_SCORE)

        # save results as json
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
        with open(
            self.results_path + "/" + self.prediction_json.split("_")[0].split("/")[-1] + "_results.json", "w"
        ) as f:
            json.dump(oc_eval_results, f, indent=2)

        print("results saved to: ", self.results_path)


if __name__ == "__main__":
    args = get_args()
    evaluator = Evaluate(args.prediction_file, args.dataset_path, args.results_path, args.split, args.seed)
    evaluator.task1_evaluation() if args.task == "task1" else evaluator.task2_evaluation()
