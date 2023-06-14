import json
import random
from typing import List

import flair
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from flair.data import Sentence
from k_means_constrained import KMeansConstrained
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from transformers import enable_full_determinism

# add your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "..."


# function to get contextual embeddings from Flair Library
def get_embeddings(embeddings, sentences):
    sentences_copy = lower_case(sentences.copy())
    sent = Sentence(sentences_copy)
    embeddings.embed(sent)
    return torch.stack([token.embedding for token in sent.tokens]).float()


# function to get classic embeddings from FLair Library
def get_embeddings_static(embeddings, sentences):
    lst_embeddings = []
    sentences_copy = lower_case(sentences.copy())
    for token in sentences_copy:
        sent = Sentence(token)
        embeddings.embed(sent)
        lst_embeddings.append(sent.embedding)
    return torch.stack(lst_embeddings).float()


# set seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    flair.set_seed(seed)
    enable_full_determinism(seed=seed)


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
    plt.xlabel("Tensor 2")
    plt.ylabel("Tensor 1")
    plt.title("Similarity Matrix")
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
    tsne = TSNE(n_components=n_components, random_state=seed, init="pca", perplexity=3)
    return tsne


def load_kpca(n_components=2, seed=42):
    # Load KernelPCA
    kernel_pca = KernelPCA(n_components=n_components, kernel="poly", degree=2, random_state=seed)
    return kernel_pca


# Initialize Kmeans constrained
def load_clf(seed=42):
    clf = KMeansConstrained(n_clusters=4, size_min=4, size_max=4, random_state=seed)
    return clf


# functions useful for new script and dataset
def load_hf_dataset(dataset_path):
    dataset = load_dataset(
        "json",
        data_files={
            "train": dataset_path + "train.json",
            "validation": dataset_path + "validation.json",
            "test": dataset_path + "test.json",
        },
        field="dataset",
    )
    return dataset


def load_prediction(prediction_json):
    with open(prediction_json) as f:
        data = json.load(f)
    return data


def get_clusters(clf_embeds_final, wall_1):
    lst_words = wall_1
    lst_groups = []
    for i in range(4):
        lst_groups.append([])
    for i in range(len(clf_embeds_final)):
        lst_groups[clf_embeds_final[i]].append(lst_words[i])
    return lst_groups


def clue2group(lst_words, wall1_default):
    dict_bbb = {}
    lst_default = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    lst_replaced = [i for i in range(4, 20)]
    for i in range(len(wall1_default)):
        dict_bbb[wall1_default[i]] = lst_default[i]
    lst_words_new = lst_words.copy()
    for i in lst_words_new:
        if i in dict_bbb.keys():
            lst_words_new[lst_words_new.index(i)] = dict_bbb[i]
        else:
            lst_words_new[lst_words_new.index(i)] = lst_replaced.pop(0)
    return lst_words_new


# find wall in prediction files based on wall unique id
def find_wall(wall_id, preds):
    for i in preds:
        if i["wall_id"] == wall_id:
            return i


# check number of matches in two lists
def check_equal(gt_groups: List, pred_groups: List):
    count = 0
    for i in gt_groups:
        if i in pred_groups:
            count += 1
    return count


def get_number_of_solved_groups(gt_groups: List, pred_groups: List, debug=False):
    count = 0
    for y in gt_groups:
        matches = list(map(lambda y_hat_i: set(y).intersection(y_hat_i), pred_groups))
        if any([len(x) == 4 for x in matches]):
            count += 1
            if debug:
                print(f"Matches for {y} = {matches}")
    return count


# slice a list into n equal sublists
def slice_list(lst, n):
    return [lst[i : i + n] for i in range(0, n * n, n)]


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
