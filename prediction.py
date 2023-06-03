from flair.embeddings import ELMoEmbeddings, WordEmbeddings, BytePairEmbeddings, \
    StackedEmbeddings, TransformerWordEmbeddings, TransformerDocumentEmbeddings
from sentence_transformers import SentenceTransformer
from utils import *
from tqdm.auto import tqdm
from evaluate_only_connect import Evaluate
from arguments import get_args
import random


class ModelPrediction:
    def __init__(self, contextual=False, model_name='elmo', dataset_path='./',
                 predictions_path='./predictions/task1/', split="test", plot='none', dim_reduction='tsne', seed=42):
        self.contextual = contextual
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.predictions_path = predictions_path
        self.split = split
        self.plot = plot
        self.dim_reduction = dim_reduction
        self.seed = seed
        self.DATASET = load_hf_dataset(self.dataset_path)

    def load_model(self):
        set_seed(seed=self.seed)
        clf = load_clf(seed=self.seed)
        try:
            if self.model_name == 'elmo':
                spare_model = ELMoEmbeddings('large')
            # elif self.model_name == 'glove':
            #     spare_model = WordEmbeddings(self.model_name)
            elif self.model_name in ['glove', 'crawl', 'news']:
                spare_model = StackedEmbeddings(
                                [
                                    # standard FastText word embeddings for English
                                    WordEmbeddings(self.model_name),
                                    # Byte pair embeddings for English
                                    BytePairEmbeddings('en'),
                                ]
                            )
            else:
                if self.contextual:
                    spare_model = TransformerWordEmbeddings(self.model_name)
                else:
                    spare_model = TransformerDocumentEmbeddings(self.model_name)
        except:
            raise Exception('Model is not supported')
        return spare_model, clf


    def prediction(self, spare_model, clf, shuffle_seed=None):
        lst_oov = []
        oc_results = []
        for wall in tqdm(self.DATASET[self.split]):
            if isinstance(shuffle_seed, int):
                wall['words'] = random.Random(shuffle_seed).sample(wall['words'], len(wall['words']))
            # step 1 => get model's contextual embeddings
            if self.contextual or self.model_name  == 'elmo':
                wall_embed = get_embeddings(spare_model, wall['words'])
                if self.model_name == 'elmo' and not self.contextual:
                    # first 1024 embeddings of elmo are static
                    wall_embed = wall_embed[:, :1024]
            elif self.model_name in ['glove', 'crawl', 'news']:
                wall_embed = get_embeddings_classic(spare_model, wall['words'])
            else:
                wall_embed = get_embeddings_static(spare_model, wall['words'])
            # optional step: find number of oov words
            # cnt_oov = 0
            # for i in wall_embed:
            #     if torch.count_nonzero(i) == 0:
            #         cnt_oov += 1
            # lst_oov.append(cnt_oov)
            # step 2 => perform constrained clustering
            clf_embeds = clf.fit_predict(wall_embed.detach().cpu())
            # Optional Step: plot the clusters
            if self.plot == 'all' or self.plot == wall['wall_id']:
                plot_wall(self.model_name, wall_embed, wall, clf_embeds, self.seed, dim_reduction=self.dim_reduction)
            # step 3 => get the clusters
            predicted_groups = get_clusters(clf_embeds, wall['words'])
            wall_json = {'wall_id': wall['wall_id'], 'predicted_groups': predicted_groups}
            oc_results.append(wall_json)
        # print('\n total oov words: {} out of {}'.format(sum(lst_oov), len(lst_oov)*16))
        # print('\n')
        # print(lst_oov)
        # save results as json
        if not os.path.exists(self.predictions_path):
            os.makedirs(self.predictions_path)
        model_saved_name = self.model_name.replace('/', '-') + '-seed' + str(shuffle_seed) + '_predictions.json'
        if self.contextual:
            model_saved_name = model_saved_name.replace('-seed', '-contextual-seed')
        with open(self.predictions_path + model_saved_name, 'w') as f:
            json.dump(oc_results, f)
        print('\n predictions saved to: ', self.predictions_path)

    def average_prediction(self, number_of_runs=16):
        model_saved_name = self.model_name.replace('/', '-')
        if self.contextual:
            model_saved_name = model_saved_name + '-contextual'
        self.predictions_path = self.predictions_path + model_saved_name + '/'
        spare_model, clf = self.load_model()
        for i in tqdm(range(number_of_runs)):
            self.prediction(spare_model, clf, shuffle_seed=i)

if __name__ == '__main__':
    args = get_args()
    #### the model_name should be from huggingface model hub or in ['elmo', 'glove', 'crawl', 'news'] ###
    # ModelPrediction(args.contextual, args.model_name, args.dataset_path, args.predictions_path,
    #                 args.split, args.plot, args.dim_reduction, args.seed).prediction()
    # Evaluate(args.predictions_path + args.model_name.replace('/', '-') + '_predictions.json'
    #          , args.dataset_path, args.results_path, args.split, args.seed).task1_grouping_evaluation(shuffle_seed=0)
    ModelPrediction(args.contextual, args.model_name, args.dataset_path, args.predictions_path,
                    args.split, args.plot, args.dim_reduction, args.seed).average_prediction()
    path = args.predictions_path + args.model_name.replace('/', '-')
    if args.contextual:
        path = path + '-contextual'
    Evaluate(args.prediction_file, args.dataset_path, args.results_path, args.split,
             args.seed).task1_grouping_evaluation_batch(path)
