from flair.embeddings import ELMoEmbeddings
from flair.embeddings import TransformerWordEmbeddings
from utils import *
import warnings
from tqdm.auto import tqdm
from evaluate import Evaluate
from arguments import get_args

class ModelPrediction:
    def __init__(self, model_name='elmo', dataset_path='./', predictions_path='./predictions/', split="test", seed=42):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.predictions_path = predictions_path
        self.split = split
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
        else:
            try:
                spare_model = TransformerWordEmbeddings(self.model_name, layers='-1, -2, -3, -4')
            except:
                raise Exception('Model is not supported')

        for wall in tqdm(dataset[self.split]):
            # step 1 => get model's contextual embeddings
            wall_embed = get_embeddings(spare_model, wall['words'])
            # step 2 => perform constrained clustering
            clf_embeds = clf.fit_predict(wall_embed.cpu())
            # step 3 => get the clusters
            predicted_groups = get_clusters(clf_embeds, wall)
            wall_json = {'wall_id': wall['wall_id'], 'predicted_groups': predicted_groups}
            oc_results.append(wall_json)

        # save results as json
        if not os.path.exists(self.predictions_path):
            os.makedirs(self.predictions_path)
        with open(self.predictions_path + '/' + self.model_name.replace('/', '-') + '_predictions.json', 'w') as f:
            json.dump(oc_results, f)

        print('predictions saved to: ', self.predictions_path)


if __name__ == '__main__':
    args = get_args()
    # the model_name should be from huggingface model hub
    ModelPrediction(args.model_name, args.dataset_path, args.predictions_path, args.split, args.seed).prediction()
    Evaluate(args.predictions_path + args.model_name.replace('/', '-') + '_predictions.json'
             , args.dataset_path, args.results_path, args.split, args.seed).task1_evaluation()
