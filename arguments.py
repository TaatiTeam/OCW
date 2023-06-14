import argparse

"""
arguments for the OC Dataset
"""


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--contextual", action="store_true", help="render the demo")
    parse.add_argument(
        "--model-name", type=str, default="elmo", help="the model name either elmo or HF model"
    )
    parse.add_argument("--dataset-path", type=str, default="./OCW/", help="Path to the OC dataset")
    parse.add_argument(
        "--predictions-path", type=str, default="./predictions/", help="Path to predictions folder"
    )
    parse.add_argument(
        "--results-path", type=str, default="./results/", help="Path to results folder"
    )
    parse.add_argument("--split", type=str, default="test", help="Which split to evaluate on")
    parse.add_argument(
        "--task",
        type=str,
        default="task1-grouping",
        choices=["task1-grouping", "task2-connections"],
        help="Which task to evaluate on. Select from task1 (groups) or task2 (connections)",
    )
    parse.add_argument("--wall-id", type=str, default="none", help="Which wall id to plot.")
    parse.add_argument(
        "--dim-reduction",
        type=str,
        default="tsne",
        choices=["tsne", "pca", "kernel_pca"],
        help="Which dimensionality reduction method to use for plotting. Select from tsne, pca, or kernel_pca",
    )
    parse.add_argument("--seed", type=int, default=42, help="the random seeds")
    parse.add_argument("--shuffle-seed", type=int, default=0, help="the random seeds")
    parse.add_argument(
        "--prediction-file", type=str, default="none", help="Path to predictions file"
    )

    args = parse.parse_args()

    return args
