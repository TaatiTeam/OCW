from utils import *
from tqdm.auto import tqdm
from sklearn.metrics.cluster import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from arguments import get_args

class Evaluate:
    def __init__(self, prediction_file, dataset_path='./', results_path='./results', split="test", seed=42):
        self.prediction_json = prediction_file
        self.dataset_path = dataset_path
        self.results_path = results_path
        self.split = split
        self.seed = seed

        self.ARI = []
        self.NMI = []
        self.FULL_WALL = 0
        self.CORRECT_GROUPS = 0

    def task1_evaluation(self):
        set_seed(seed=self.seed)
        oc_eval_results = {'global': {}, 'granular': []}
        if not os.path.exists(self.dataset_path):
            raise Exception('Dataset path does not exist')
        dataset = load_hf_dataset(self.dataset_path)
        prediction = load_prediction(self.prediction_json)
        for wall in tqdm(dataset[self.split]):
            gt_words = [i['gt_words'] for i in wall['groups'].values()]
            pred_words = find_wall(wall['wall_id'], prediction)
            gt_sorted = [sorted(i) for i in gt_words]
            pred_sorted = [sorted(i) for i in pred_words]
            correct_groups = check_equal(pred_sorted, gt_sorted)
            self.CORRECT_GROUPS += correct_groups
            if correct_groups == 4:
                self.FULL_WALL += 1
            gt_lst = [item for sublist in gt_sorted for item in sublist]
            pred_lst = [item for sublist in pred_sorted for item in sublist]
            index_gt = clue2group(gt_lst, gt_lst)
            index_pred = clue2group(pred_lst, gt_lst)
            ari_val = ari(index_pred, index_gt)
            if ari_val < 0:
                ari_val = 0
            nmi_val = nmi(index_pred, index_gt)
            self.ARI.append(ari_val)
            self.NMI.append(nmi_val)
            oc_eval_results['granular'].append({'wall_id': wall['wall_id'], 'ARI': ari_val, 'NMI': nmi_val
                                                , 'correct_groups': correct_groups
                                                , 'full_wall': correct_groups == 4})
        oc_eval_results['global']['ARI'] = np.mean(self.ARI)
        oc_eval_results['global']['NMI'] = np.mean(self.NMI)
        oc_eval_results['global']['full_wall'] = self.FULL_WALL
        oc_eval_results['global']['correct_groups'] = self.CORRECT_GROUPS

        # save results as json
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
        with open(self.results_path + '/' + self.prediction_json.split('_')[0].split('/')[-1] + '_results.json', 'w') as f:
            json.dump(oc_eval_results, f)

        print('results saved to: ', self.results_path)


if __name__ == '__main__':
    args = get_args()
    Evaluate(args.prediction_file, args.dataset_path, args.results_path, args.seed).task1_evaluation()
