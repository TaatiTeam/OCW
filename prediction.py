from flair.embeddings import ELMoEmbeddings
from flair.embeddings import TransformerWordEmbeddings
from utils import *
import warnings
from tqdm.auto import tqdm
from evaluate_only_connect import Evaluate
from arguments import get_args
from sentence_transformers import SentenceTransformer
import gensim.downloader

class ModelPrediction:
    def __init__(self, model_name='elmo', dataset_path='./',
                 predictions_path='./predictions/task1/', split="test", plot='none', seed=42):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.predictions_path = predictions_path
        self.split = split
        self.plot = plot
        self.seed = seed

    def prediction(self):
        warnings.filterwarnings("ignore")
        set_seed(seed=self.seed)
        clf = load_clf(seed=self.seed)
        if not os.path.exists(self.dataset_path):
            raise Exception('Dataset path does not exist')
        dataset = load_hf_dataset(self.dataset_path)
        oc_results = []
        if self.model_name == 'elmo':
            spare_model = ELMoEmbeddings(embedding_mode='top')
        elif self.model_name == 'glove':
            spare_model = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.840B.300d')
        elif self.model_name == 'wv':
            spare_model = gensim.downloader.load('word2vec-google-news-300')
        else:
            try:
                spare_model = TransformerWordEmbeddings(self.model_name, layers='-1, -2, -3, -4')
            except:
                raise Exception('Model is not supported')
        # cnt_wrd = 0
        for wall in tqdm(dataset[self.split]):
            # step 1 => get model's contextual embeddings
            if self.model_name == 'glove':
                wall_embed = get_embeddings_glove(spare_model, wall['words'])
                # cnt_wrd += get_embeddings_glove(spare_model, wall['words'])[1]
            elif self.model_name == 'wv':
                wall_embed = get_embeddings_wv(spare_model, wall['words'])[0]
                # cnt_wrd += get_embeddings_wv(spare_model, wall['words'])[1]
            else:
                wall_embed = get_embeddings(spare_model, lower_case(wall['words']))
            # step 2 => perform constrained clustering
            clf_embeds = clf.fit_predict(wall_embed.cpu())
            # Optional Step: plot the clusters
            if self.plot == 'all':
                plot_wall(self.model_name, wall_embed, wall, clf_embeds, dim_reduction='pca')
            elif self.plot == wall['wall_id']:
                plot_wall(self.model_name, wall_embed, wall, clf_embeds, dim_reduction='pca')
            # step 3 => get the clusters
            predicted_groups = get_clusters(clf_embeds, wall)
            wall_json = {'wall_id': wall['wall_id'], 'predicted_groups': predicted_groups}
            oc_results.append(wall_json)
        # print('total words not in w2v: ', cnt_wrd)

        # save results as json
        if not os.path.exists(self.predictions_path):
            os.makedirs(self.predictions_path)
        with open(self.predictions_path + '/' + self.model_name.replace('/', '-') + '_predictions.json', 'w') as f:
            json.dump(oc_results, f)

        print('predictions saved to: ', self.predictions_path)


if __name__ == '__main__':
    args = get_args()
    # the model_name should be from huggingface model hub
    ModelPrediction(args.model_name, args.dataset_path, args.predictions_path, args.split, args.plot, args.seed).prediction()
    Evaluate(args.predictions_path + args.model_name.replace('/', '-') + '_predictions.json'
             , args.dataset_path, args.results_path, args.split, args.seed).task1_grouping_evaluation()
