import json
import os
import random

import utils as ocw_utils
from arguments import get_args
from evaluate_only_connect import Evaluate
from flair.embeddings import (
    BytePairEmbeddings,
    DocumentPoolEmbeddings,
    ELMoEmbeddings,
    TransformerDocumentEmbeddings,
    TransformerWordEmbeddings,
    WordEmbeddings,
)
from tqdm.auto import tqdm


class ModelPrediction:
    def __init__(
        self,
        contextual=False,
        model_name="intfloat/e5-base-v2",
        dataset_path="../dataset/",
        predictions_path="../predictions/task1/",
        split="test",
        seed=42,
    ):
        self.contextual = contextual
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.predictions_path = predictions_path
        self.split = split
        self.seed = seed
        self.DATASET = ocw_utils.load_hf_dataset(self.dataset_path)

    def load_model(self):
        ocw_utils.set_seed(seed=self.seed)
        clf = ocw_utils.load_clf(seed=self.seed)
        try:
            if self.model_name == "elmo":
                spare_model = ELMoEmbeddings("large")

            elif self.model_name in ["glove", "crawl", "news"]:
                # embedding that performs mean pooling by default
                # Byte pair embeddings to handle OOV words
                spare_model = DocumentPoolEmbeddings(
                    [WordEmbeddings(self.model_name), BytePairEmbeddings("en")]
                )

            else:
                if self.contextual:
                    spare_model = TransformerWordEmbeddings(self.model_name)
                else:
                    spare_model = TransformerDocumentEmbeddings(self.model_name)
        except ValueError:
            raise Exception("Model is not supported")
        return spare_model, clf

    def prediction(self, spare_model, clf, shuffle_seed=None):
        # lst_oov = []
        oc_results = []
        for wall in tqdm(self.DATASET[self.split]):
            if isinstance(shuffle_seed, int):
                wall["words"] = random.Random(shuffle_seed).sample(
                    wall["words"], len(wall["words"])
                )
            # step 1 => get model's embeddings
            if self.contextual or self.model_name == "elmo":
                wall_embed = ocw_utils.get_embeddings(spare_model, wall["words"])
                if self.model_name == "elmo" and not self.contextual:
                    # first 1024 embeddings of elmo are static
                    wall_embed = wall_embed[:, :1024]
            else:
                wall_embed = ocw_utils.get_embeddings_static(spare_model, wall["words"])
            # optional step: find number of oov words
            # cnt_oov = 0
            # for i in wall_embed:
            #     if torch.count_nonzero(i) == 0:
            #         cnt_oov += 1
            # lst_oov.append(cnt_oov)
            # step 2 => perform constrained clustering
            clf_embeds = clf.fit_predict(wall_embed.detach().cpu())
            # Optional Step: plot the clusters
            if self.plot == "all" or self.plot == wall["wall_id"]:
                ocw_utils.plot_wall(
                    self.model_name,
                    wall_embed,
                    wall,
                    clf_embeds,
                    self.seed,
                    dim_reduction=self.dim_reduction,
                )
            # step 3 => get the clusters
            predicted_groups = ocw_utils.get_clusters(clf_embeds, wall["words"])
            wall_json = {
                "wall_id": wall["wall_id"],
                "predicted_groups": predicted_groups,
                "predicted_connections": None,
                "gt_groups": [
                    wall["groups"]["group_1"]["gt_words"],
                    wall["groups"]["group_2"]["gt_words"],
                    wall["groups"]["group_3"]["gt_words"],
                    wall["groups"]["group_4"]["gt_words"],
                ],
                "gt_connections": wall["gt_connections"],
            }
            oc_results.append(wall_json)
        # print('\n total oov words: {} out of {}'.format(sum(lst_oov), len(lst_oov)*16))
        # print('\n')
        # print(lst_oov)
        # save results as json
        if not os.path.exists(self.predictions_path):
            os.makedirs(self.predictions_path)
        model_saved_name = (
            self.model_name.replace("/", "-") + "-seed" + str(shuffle_seed) + "_predictions.json"
        )
        if self.contextual:
            model_saved_name = model_saved_name.replace("-seed", "-contextual-seed")
        with open(self.predictions_path + model_saved_name, "w") as f:
            json.dump(oc_results, f, ensure_ascii=False, indent=2)
        print("\n predictions saved to: ", self.predictions_path)

    def average_prediction(self, number_of_runs=16):
        model_saved_name = self.model_name.replace("/", "-")
        if self.contextual:
            model_saved_name = model_saved_name + "-contextual"
        self.predictions_path = self.predictions_path + model_saved_name + "/"
        spare_model, clf = self.load_model()
        for i in tqdm(range(number_of_runs)):
            self.prediction(spare_model, clf, shuffle_seed=i)


if __name__ == "__main__":
    args = get_args()
    # the model_name should be from huggingface model hub or in ['elmo', 'glove', 'crawl', 'news']
    ModelPrediction(
        args.contextual,
        args.model_name,
        args.dataset_path,
        args.predictions_path,
        args.split,
        args.seed,
    ).average_prediction()
    path = args.predictions_path + args.model_name.replace("/", "-")
    if args.contextual:
        path = path + "-contextual"
    Evaluate(
        args.prediction_file, path, args.dataset_path, args.results_path, args.split, args.seed
    ).task1_grouping_evaluation_batch()
