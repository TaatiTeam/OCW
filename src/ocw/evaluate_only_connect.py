import json
import os

import evaluate
import numpy as np
from evaluate import load
from scipy.stats import wasserstein_distance as wd
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import fowlkes_mallows_score as fms
from tqdm.auto import tqdm

import ocw.utils as ocw_utils
from ocw.arguments import get_args

evaluate.logging.set_verbosity_error()


class Evaluate:
    def __init__(
        self,
        prediction_file=None,
        prediction_path=None,
        dataset_path="../dataset/",
        results_path="../results/",
        split="test",
        seed=42,
    ):
        self.prediction_file = prediction_file
        self.prediction_path = prediction_path
        self.dataset_path = dataset_path
        self.results_path = results_path
        self.split = split
        self.seed = seed
        self.DATASET = ocw_utils.load_hf_dataset(self.dataset_path)

        # Task 1 - grouping metrics
        self.WD = []
        self.AMI = []
        self.FULL_WALL = 0
        self.CORRECT_GROUPS = 0
        # Task 1 - additional metrics
        self.FMS = []
        self.ARI = []

        # Task 2 - connections metrics
        self.EXACT_MATCH = []
        self.ROUGE = []
        self.BERT_SCORE = []

    def task1_grouping_evaluation(self):
        ocw_utils.set_seed(seed=self.seed)
        oc_eval_results = {"global": {}, "granular": []}
        if not os.path.exists(self.prediction_file):
            raise Exception("prediction file does not exist")
        prediction = ocw_utils.load_prediction(self.prediction_file)
        for wall in tqdm(self.DATASET[self.split]):
            gt_words = [i["gt_words"] for i in wall["groups"].values()]
            pred_words = ocw_utils.find_wall(wall["wall_id"], prediction)["predicted_groups"]
            gt_sorted = [sorted(ocw_utils.lower_case(i)) for i in gt_words]
            pred_sorted = [sorted(ocw_utils.lower_case(i)) for i in pred_words]
            correct_groups = ocw_utils.check_equal(gt_sorted, pred_sorted)
            self.CORRECT_GROUPS += correct_groups
            if correct_groups == 4:
                self.FULL_WALL += 1
            gt_lst = [item for sublist in gt_sorted for item in sublist]
            pred_lst = [item for sublist in pred_sorted for item in sublist]
            index_gt = ocw_utils.clue2group(gt_lst, gt_lst)
            index_pred = ocw_utils.clue2group(pred_lst, gt_lst)
            ami_val = ami(index_gt, index_pred)
            # additional metrics
            ari_val = ari(index_gt, index_pred)
            fms_val = fms(index_gt, index_pred)
            pred_sliced = ocw_utils.slice_list(index_pred, 4)
            gt_sliced = ocw_utils.slice_list(index_gt, 4)
            wd_val = 0
            pred_sliced, gt_sliced = ocw_utils.remove_same(pred_sliced, gt_sliced)
            for i in range(len(pred_sliced)):
                wd_val += min(wd(pred_sliced[i], gt_sliced[i]), 1)

            # normalize wd to be in [0, 1]
            self.WD.append(wd_val / 4)
            self.AMI.append(ami_val)
            # additional metrics
            self.ARI.append(ari_val)
            self.FMS.append(fms_val)

            oc_eval_results["granular"].append(
                {
                    "wall_id": wall["wall_id"],
                    "WD": wd_val / 4,
                    "AMI": ami_val,
                    "ARI": ari_val,
                    "FMS": fms_val,
                    "correct_groups": correct_groups,
                    "full_wall": correct_groups == 4,
                }
            )
        oc_eval_results["global"]["WD"] = np.mean(self.WD)
        oc_eval_results["global"]["AMI"] = np.mean(self.AMI)
        # additional metrics
        oc_eval_results["global"]["ARI"] = np.mean(self.ARI)
        oc_eval_results["global"]["FMS"] = np.mean(self.FMS)

        oc_eval_results["global"]["full_wall"] = self.FULL_WALL
        oc_eval_results["global"]["correct_groups"] = self.CORRECT_GROUPS

        # save results as json
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
        with open(
            self.results_path
            + self.prediction_file.split("_")[-2].split("/")[-1]
            + "_results.json",
            "w",
        ) as f:
            json.dump(oc_eval_results, f, indent=2)

        print("\n results saved to: ", self.results_path)

    def task1_grouping_evaluation_batch(self):
        json_files = [
            pos_json
            for pos_json in os.listdir(self.prediction_path)
            if pos_json.endswith("_predictions.json")
        ]
        self.results_path = self.results_path + self.prediction_path.split("/")[-1] + "/"
        for file in tqdm(json_files):
            self.prediction_file = self.prediction_path + "/" + file
            self.WD = []
            self.AMI = []
            self.FULL_WALL = 0
            self.CORRECT_GROUPS = 0
            # Task 1 - additional metrics
            self.FMS = []
            self.ARI = []
            self.task1_grouping_evaluation()
        results_files = [
            pos_json
            for pos_json in os.listdir(self.results_path)
            if pos_json.endswith("_results.json")
        ]
        results = {}
        wd_list = []
        ami_list = []
        # additional metrics
        ari_list = []
        fms_list = []

        full_wall_list = []
        correct_groups_list = []
        for file in results_files:
            with open(self.results_path + "/" + file) as f:
                data = json.load(f)
                wd_list.append(data["global"]["WD"])
                ami_list.append(data["global"]["AMI"])
                # additional metrics
                ari_list.append(data["global"]["ARI"])
                fms_list.append(data["global"]["FMS"])

                full_wall_list.append(data["global"]["full_wall"])
                correct_groups_list.append(data["global"]["correct_groups"])
        results["mean_wd"] = np.mean(wd_list)
        results["std_wd"] = np.std(wd_list)
        results["mean_ami"] = np.mean(ami_list)
        results["std_ami"] = np.std(ami_list)
        # additional metrics
        results["mean_ari"] = np.mean(ari_list)
        results["std_ari"] = np.std(ari_list)
        # results["ari_list"] = ari_list
        results["mean_fms"] = np.mean(fms_list)
        results["std_fms"] = np.std(fms_list)

        results["mean_full_wall"] = np.mean(full_wall_list)
        results["std_full_wall"] = np.std(full_wall_list)
        results["mean_correct_groups"] = np.mean(correct_groups_list)
        results["std_correct_groups"] = np.std(correct_groups_list)
        with open(self.results_path + "/batch_output.json", "w") as f:
            json.dump(results, f, indent=2)

    def task2_connections_evaluation(self):
        ocw_utils.set_seed(seed=self.seed)
        exact_match = load("exact_match")
        rouge = load("rouge")
        bertscore = load("bertscore")
        oc_eval_results = {"global": {}, "granular": []}
        if not os.path.exists(self.prediction_file):
            raise Exception("prediction file does not exist")
        prediction = ocw_utils.load_prediction(self.prediction_file)
        for wall in tqdm(self.DATASET[self.split]):
            gt_connections = wall["gt_connections"]
            pred_connections = ocw_utils.find_wall(wall["wall_id"], prediction)[
                "predicted_connections"
            ]
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
            self.results_path + self.prediction_file.split("_")[0].split("/")[-1] + "_results.json",
            "w",
        ) as f:
            json.dump(oc_eval_results, f, indent=2)
        print("results saved to: ", self.results_path)


if __name__ == "__main__":
    args = get_args()
    evaluator = Evaluate(
        args.prediction_file,
        args.predictions_path,
        args.dataset_path,
        args.results_path,
        args.split,
        args.seed,
    )
    if args.task == "task1-grouping":
        if args.prediction_file.endswith(".json"):
            evaluator.task1_grouping_evaluation()
        else:
            evaluator.task1_grouping_evaluation_batch()
    else:
        evaluator.task2_connections_evaluation()
