from datasets import load_dataset
import matplotlib.pyplot as plt
from flair.data import Sentence
import flair
import torch
from k_means_constrained import KMeansConstrained
import random
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
from scipy import interpolate
import os
import re
import numpy as np
import json

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# add your OpenAI API key
# openai.api_key = open("key.txt", "r").read().strip('\n')

# function to get embeddings of a string from Flair Library
def get_embeddings(embeddings, sentence):
    sent = Sentence(sentence)
    embeddings.embed(sent)
    return torch.stack(
        [token.embedding for token in sent.tokens]
    ).float()

# function to get Glove embeddings
def get_embeddings_glove(embeddings, sentence):
    return torch.stack(
            [torch.as_tensor(embeddings.encode(token)) for token in sentence]
        ).float()

# function to get Word2Vec embeddings
def get_embeddings_wv(embeddings, sentence):
    lst_embed = []
    cnt_zero = 0
    for token in sentence:
        lst_sub_embed = []
        lst_add = 0
        try:
            lst_embed.append(torch.as_tensor(embeddings[token]))
        except:
            cnt_zero += 1
            if len(token.split(' ')) > 1:
                token = [i for i in token.split(' ') if i not in stopwords.words('english')]
                for tokens in token:
                    try:
                        lst_sub_embed.append(embeddings[tokens])
                    except:
                        lst_sub_embed.append(np.zeros(300))
                for i in lst_sub_embed:
                    lst_add += i
                    lst_add = lst_add/len(lst_sub_embed)
                lst_embed.append(torch.as_tensor(lst_add))
            else:
                lst_embed.append(torch.zeros(300))
    return torch.stack(lst_embed).float(), cnt_zero

# set seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    flair.set_seed(seed)


# construct similarity matrix
def cosine_similarity(tensor1, tensor2):
    dot_product = torch.matmul(tensor1, tensor2.t())
    norm1 = torch.norm(tensor1, dim=1)
    norm2 = torch.norm(tensor2, dim=1)
    similarity_matrix = dot_product / torch.ger(norm1, norm2)
    return similarity_matrix


# plot similarity matrix
def plot_similarity_matrix(similarity_matrix):
    plt.imshow(similarity_matrix)
    plt.colorbar()
    plt.grid(b=None)
    plt.xlabel('Tensor 2')
    plt.ylabel('Tensor 1')
    plt.title('Similarity Matrix')
    plt.show()


# from wall and embedding to plotting similarity
def compute_plot_similarity(embedding, wall):
    wall1 = get_embeddings(embedding, wall)
    wall1_sims = cosine_similarity(wall1, wall1)
    return plot_similarity_matrix(wall1_sims.cpu().numpy())


def load_pca(n_components=2, seed=42):
    # Load PCA
    pca = PCA(n_components=n_components, random_state=seed)
    return pca


def load_tsne(n_components=2, seed=42):
    # Load T-SNE
    tsne = TSNE(n_components=n_components, random_state=seed, init='pca', perplexity=3)
    return tsne


def load_kpca(n_components=2, seed=42):
    # Load KernelPCA
    kernel_pca = KernelPCA(n_components=n_components, kernel='poly', degree=2, random_state=seed)
    return kernel_pca


# Initialize Kmeans constrained
def load_clf(seed=42):
    clf = KMeansConstrained(
        n_clusters=4,
        size_min=4,
        size_max=4,
        random_state=seed)
    return clf

def plot_wall(model_name, embeddings, wall, clusteringOutput, dim_reduction='pca', save_path='./plots/'):
    saved_file = 'model_' + model_name.replace('/', '-') + '_wall_' + wall['wall_id'] + '_' + dim_reduction + '.jpg'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if dim_reduction == 'tsne':
        reduction = load_tsne().fit_transform(embeddings.cpu())
    elif dim_reduction == 'pca':
        reduction = load_pca(n_components=2).fit_transform(embeddings.cpu())
    elif dim_reduction == 'kernel_pca':
        reduction = load_kpca().fit_transform(embeddings.cpu())
    else:
        raise ValueError('No Valid Dimentionality Reduction Was Found')

    # label = load_clf().fit_predict(reduction)
    label = clusteringOutput
    # Getting unique labels
    wall1_default = wall['groups']['group_1']['gt_words'] + wall['groups']['group_2']['gt_words']\
                    + wall['groups']['group_3']['gt_words'] + wall['groups']['group_4']['gt_words']
    connections = wall['gt_connections']
    emmbedes_lst = wall['words']
    LABEL_TRUE = clue2group(emmbedes_lst, wall1_default)
    U_LABEL_TRUE = np.unique(LABEL_TRUE)
    # centroids = clf.cluster_centers_
    colors = ['purple', 'green', 'red', 'blue']
    font = {'family': 'Times New Roman',
            'size': 7}

    plt.rc('font', **font)
    plt.figure(figsize=(7, 4))
    # plotting the results:
    j = 0
    for i in U_LABEL_TRUE:
        plt.scatter(reduction[LABEL_TRUE == i, 0], reduction[LABEL_TRUE == i, 1],
                    label='Group ' + str(i + 1) + ': ' + connections[i], s=100,
                    color=colors[i])
    for name, x, y in zip(emmbedes_lst, reduction[:, 0], reduction[:, 1]):
        plt.annotate(name, xy=(x, y), xytext=(-7, 5), textcoords='offset points')

    # Plot the centroids as a black X
    # plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='k')

    # draw enclosure
    for i in label:
        points = reduction[label == i]
        # get convex hull
        hull = ConvexHull(points)
        # get x and y coordinates
        # repeat last point to close the polygon
        x_hull = np.append(points[hull.vertices, 0],
                           points[hull.vertices, 0][0])
        y_hull = np.append(points[hull.vertices, 1],
                           points[hull.vertices, 1][0])
        # # plot shape
        # plt.fill(x_hull, y_hull, alpha=0.3, c='gainsboro')

        # interpolate
        dist = np.sqrt((x_hull[:-1] - x_hull[1:]) ** 2 + (y_hull[:-1] - y_hull[1:]) ** 2)
        dist_along = np.concatenate(([0], dist.cumsum()))
        spline, u = interpolate.splprep([x_hull, y_hull],
                                        u=dist_along, s=0)
        interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
        interp_x, interp_y = interpolate.splev(interp_d, spline)
        # plot shape
        plt.fill(interp_x, interp_y, '--', c='gainsboro', alpha=0.1)

    # plt.grid(b=None)
    plt.legend(loc='upper center')
    # plt.show()
#
    plt.savefig(save_path + saved_file, dpi=300)

### functions useful for new script and dataset ###
def load_hf_dataset(dataset_path):
    dataset = load_dataset('json', data_files={'train': dataset_path + 'train.json',
                                               'validation': dataset_path + 'validation.json',
                                               'test': dataset_path + 'test.json'}, field='dataset')
    return dataset


def load_prediction(prediction_json):
    with open(prediction_json) as f:
        data = json.load(f)
    return data


def get_clusters(clf_embeds_final, wall_1):
    lst_words = wall_1['words']
    lst_groups = []
    for i in range(4):
        lst_groups.append([])
    for i in range(len(clf_embeds_final)):
        lst_groups[clf_embeds_final[i]].append(lst_words[i])
    return lst_groups


def clue2group(lst_words, wall1_default):
    dict_bbb = {}
    lst_aaa_new = []
    lst_default = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    lst_replaced = [i for i in range (4, 20)]
    for i in range(len(wall1_default)):
        dict_bbb[wall1_default[i]] = lst_default[i]
    lst_words_new = lst_words.copy()
    # if len(set(lst_words_new)) != len(set(wall1_default)):
    #     return lst_replaced
    for i in lst_words_new:
        if i in dict_bbb.keys():
            lst_words_new[lst_words_new.index(i)] = dict_bbb[i]
        else:
            lst_words_new[lst_words_new.index(i)] = lst_replaced.pop(0)
    return lst_words_new


# find wall in prediction files based on wall unique id
def find_wall(wall_id, preds):
    for i in preds:
        if i['wall_id'] == wall_id:
            return i


# check number of matches in two lists
def check_equal(lst1, lst2):
    count = 0
    for i in lst1:
        if i in lst2:
            count += 1
    return count

# slice a list into n equal sublists
def slice_list(lst, n):
    return [lst[i:i+n] for i in range(0, n*n, n)]

# remove same items from two lists
def remove_same(lst1, lst2):
    lst1_new = lst1.copy()
    lst2_new = lst2.copy()
    for i in lst1:
        if i in lst2:
            lst1_new.remove(i)
            lst2_new.remove(i)
    return lst1_new, lst2_new

# lower case a list of words
def lower_case(lst):
    lst_new = []
    for i in lst:
        lst_new.append(i.lower())
    return lst_new

