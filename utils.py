import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flair.data import Sentence
from flair.embeddings import ELMoEmbeddings
from flair.embeddings import TransformerWordEmbeddings
import flair
import torch
from torch import nn
from k_means_constrained import KMeansConstrained
import random
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import hmean
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
from scipy import interpolate
import os
import warnings
from df_utils import *

warnings.filterwarnings("ignore")
EPISODE_ITEMS = 32
WALL_ITEMS = EPISODE_ITEMS / 2
LABEL_TRUE = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
U_LABEL_TRUE = np.unique(LABEL_TRUE)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# function to get embeddings of a string from Flair Library
def get_embeddings(embeddings, sentence):
    sent = Sentence(sentence)
    embeddings.embed(sent)
    return torch.stack(
        [token.embedding for token in sent.tokens]
    ).float()


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


def geo_mean_overflow(iterable):
    return np.exp(np.log(iterable).mean())


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


def wall_evaluation(label, label_true=LABEL_TRUE, metrics=['ars', 'rs', 'amis', 'nmis']):
    metric_dict = {}
    if not metrics or len(metrics) > 4:
        raise ValueError('metrics must be a list of length 1-4')
    if 'ars' in metrics:
        ars = adjusted_rand_score(label, label_true)
        metric_dict['ars'] = ars
    if 'rs' in metrics:
        rs = rand_score(label, label_true)
        metric_dict['rs'] = rs
    if 'amis' in metrics:
        amis = adjusted_mutual_info_score(label, label_true)
        metric_dict['amis'] = amis
    if 'nmis' in metrics:
        nmis = normalized_mutual_info_score(label, label_true)
        metric_dict['nmis'] = nmis
    return metric_dict


# evaluates a single wall
def wall_evaluator(num_default=0, wall_skip=0, embeddings=None, dim_reduction='tsne', seed=42):
    if not embeddings:
        raise ValueError('embeddings must be a Flair embedding object')
    set_seed(seed=seed)
    clf = load_clf()
    pca = load_pca()
    tsne = load_tsne()
    df_new = DataFrameUtils().preprocess_text()
    wall_info = {}
    num = (num_default * EPISODE_ITEMS) + (wall_skip * WALL_ITEMS)
    wall1_default = df_new[int(num - WALL_ITEMS):int(num)]['Names'].tolist()
    wall_info['names'] = wall1_default
    wall1_shuffled = wall1_default.copy()
    random.shuffle(wall1_shuffled)
    wall1 = " ".join(wall1_shuffled)
    wall1_embed = get_embeddings(embeddings, wall1)
    wall1_embed_sims = cosine_similarity(wall1_embed, wall1_embed)

    # method1 => clf + embeds
    clf_embeds = clf.fit_predict(wall1_embed.cpu())
    clf_embeds = clf.labels_
    clf_embeds_metrics = wall_evaluation(clf_embeds)
    wall_info['clf_embeds'] = (clf_embeds, clf_embeds_metrics)

    # method2 => clf + ssm
    clf_embeds_ssm = clf.fit_predict(wall1_embed_sims.cpu())
    clf_embeds_ssm = clf.labels_
    clf_embeds_ssm_metrics = wall_evaluation(clf_embeds_ssm)
    wall_info['clf_embeds_ssm'] = (clf_embeds_ssm, clf_embeds_ssm_metrics)

    # method3 => pca + clf + embeds
    clf_pca_embeds = clf.fit_predict(pca.fit_transform(wall1_embed.cpu()))
    clf_pca_embeds = clf.labels_
    clf_pca_embeds_metrics = wall_evaluation(clf_pca_embeds)
    wall_info['clf_pca_embeds'] = (clf_pca_embeds, clf_pca_embeds_metrics)

    # method4 => pca + clf + ssm
    clf_pca_embeds_ssm = clf.fit_predict(pca.fit_transform(wall1_embed_sims.cpu()))
    clf_pca_embeds_ssm = clf.labels_
    clf_pca_embeds_ssm_metrics = wall_evaluation(clf_pca_embeds_ssm)
    wall_info['clf_pca_embeds_ssm'] = (clf_pca_embeds_ssm, clf_pca_embeds_ssm_metrics)

    # method5 => tsne + clf + embeds
    clf_tsne_embeds = clf.fit_predict(tsne.fit_transform(wall1_embed.cpu()))
    clf_tsne_embeds = clf.labels_
    clf_tsne_embeds_metrics = wall_evaluation(clf_tsne_embeds)
    wall_info['clf_tsne_embeds'] = (clf_tsne_embeds, clf_tsne_embeds_metrics)

    # method6 => tsne + clf + ssm
    clf_tsne_embeds_ssm = clf.fit_predict(tsne.fit_transform(wall1_embed_sims.cpu()))
    clf_tsne_embeds_ssm = clf.labels_
    clf_tsne_embeds_ssm_metrics = wall_evaluation(clf_tsne_embeds_ssm)
    wall_info['clf_tsne_embeds_ssm'] = (clf_tsne_embeds_ssm, clf_tsne_embeds_ssm_metrics)

    # plot & save solved wall
    # plot_solved_wall(wall1_embed, wall_skip, wall1_default, dim_reduction=dim_reduction, wall_info= wall_info)

    return wall_info


def plot_solved_wall(embeddings, wall_skip, wall1_default, wall_info, dim_reduction='tsne', saved_path=None):
    if not saved_path:
        saved_path = './nlp_plots/'
        saved_file = 'wall' + str(wall1_default) + str(wall_skip) + '_' + dim_reduction + '.jpg'
    if not os.path.isdir(saved_path):
        os.mkdir(saved_path)
    if dim_reduction == 'tsne':
        df_tsne = load_tsne().fit_transform(embeddings.cpu())
    elif dim_reduction == 'pca':
        df_tsne = load_pca().fit_transform(embeddings.cpu())
    elif dim_reduction == 'kernel_pca':
        df_tsne = load_kpca().fit_transform(embeddings.cpu())
    else:
        raise ValueError('No Valid Dimentionality Reduction Was Found')

    # predict the labels of clusters.
    # label = clf.fit_predict(df_tsne)
    # label = clf.fit_predict(wall1_elmo_sims.cpu())
    # label = clf.labels_
    wall_info = wall_evaluator(wall_skip=wall_skip, embeddings=embeddings, dim_reduction=dim_reduction, seed=42)
    label = wall_info['clf_embeds'][0]

    # Getting unique labels
    u_labels = np.unique(label)

    emmbedes_lst = wall1_default
    # centroids = clf.cluster_centers_
    # colors = ['purple','purple','purple','purple', 'green','green','green','green', 'red','red','red','red', 'blue','blue','blue','blue']
    colors = ['purple', 'green', 'red', 'blue']
    # df_tsne['KMeans_cluster_constrained'][start_len:start_len+16] = clf.labels_
    plt.figure(figsize=(14, 8))
    # plotting the results:
    j = 0
    for i in U_LABEL_TRUE:
        plt.scatter(df_tsne[LABEL_TRUE == i, 0], df_tsne[LABEL_TRUE == i, 1], label='Group ' + str(i + 1), s=100,
                    color=colors[i])
    for name, x, y in zip(emmbedes_lst, df_tsne[:, 0], df_tsne[:, 1]):
        plt.annotate(name, xy=(x, y), xytext=(-15, 10), textcoords='offset points')

    # Plot the centroids as a black X
    # plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='k')

    # draw enclosure
    for i in label:
        points = df_tsne[label == i]
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

    plt.grid(b=None)
    plt.legend()
    # plt.show()

    plt.savefig(saved_path + saved_file, dpi=300)


# removes flagged season for one-season-out cross validation
def remove_season(dataset_json_original, season_flag):
    dataset_json = dataset_json_original.copy()
    if season_flag < 1 or season_flag > 15:
        raise ValueError('Season Not Available')
    series_dict_accumulated = DataFrameUtils().accumulated_series_dict()
    prev_index = series_dict_accumulated[season_flag-1] * ITEM_NUM
    cur_index = series_dict_accumulated[season_flag] * ITEM_NUM
    val_json = dataset_json_original[prev_index:cur_index]
    del dataset_json[prev_index:cur_index]
    return dataset_json, val_json


def create_dataset_json(path, dataset_json_original, num_seasons=15):
    if not os.path.exists(path):
        os.makedirs(path)
    if num_seasons < 1 or num_seasons > 15:
        raise ValueError('Season Not Available')
    for season in range(1, num_seasons+1):
        train_lst, val_lst = remove_season(dataset_json_original, season)
        # # save train dataset as jsonl
        with open(path + '/' + 'onlyconnect_triplet_train_season_' + str(season) + '.jsonl', 'w') as outfile:
            for entry in train_lst:
                json.dump(entry, outfile)
                outfile.write('\n')
        outfile.close()
        # # save validation dataset as jsonl
        with open(path + '/' + 'onlyconnect_triplet_val_season_' + str(season) + '.jsonl', 'w') as outfile:
            for entry in val_lst:
                json.dump(entry, outfile)
                outfile.write('\n')
        outfile.close()
        print("json datasets successfully created!")