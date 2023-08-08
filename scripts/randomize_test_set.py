import copy
import json
import random
import shutil
from pathlib import Path

import typer
from rich import print
from rich.progress import track
from typing_extensions import Annotated

import ocw.utils as ocw_utils
from ocw.common import TEST_FN, TRAIN_FN, VALID_FN

# Set the random seed for reproducibility
random.seed(42)


def main(
    path: Annotated[Path, typer.Argument(help="Path to the OCW dataset on disk")],
    output_dir: Annotated[Path, typer.Argument(help="Path to the output directory")],
):
    """Make a copy of the dataset with a randomized test set. Train and valid sets are unchanged.

    We attempt to remove red herrings by replacing the second, third and fourth group of each wall with a group
    sampled from a different wall, the intuition being that walls constructed from randomly chosen groups from
    four different walls are unlikely to contain red herrings. The new groups are chosen such that the resulting
    wall doesn't contain any duplicate clues, and the groups of each wall are from different walls.

    Args:
        path (str): Path to the OCW dataset on disk.
    """
    # Load and make a copy of the dataset
    path = Path(path)
    ocw_test = json.load(open(path / TEST_FN, "r"))
    season_to_walls_map = ocw_test["season_to_walls_map"]
    filtered_test = copy.deepcopy(ocw_test["dataset"])
    print(f"[green]:white_check_mark: Loaded test set from {(path / TEST_FN).absolute()}[/green]")

    # Get the set of all wall ids
    wall_ids = {example["wall_id"] for example in ocw_test["dataset"]}

    # Get the group keys, starting from the second (we will leave the first group as is)
    group_keys = [f"group_{i}" for i in range(2, 5)]

    # Use the same group only once for each new wall
    seen_groups = set()

    for i, example in track(
        enumerate(ocw_test["dataset"]),
        description="Filtering red-herrings from test set",
        total=len(ocw_test["dataset"]),
    ):
        # Sample groups from a wall only once for each new wall
        seen_walls = {example["wall_id"]}

        # Keep track of the words that already exist in the group
        seen_words = set(example["groups"]["group_1"]["gt_words"])

        for group_key in group_keys:
            # Find a new candidate group for the current wall that...
            while True:
                # ...doesn't belong to a wall already sampled for the current wall
                # NOTE: The sorting is expensive but necessary to ensure reproducibility
                wall_id = random.choice(sorted(list(wall_ids - seen_walls)))
                wall = ocw_utils.find_wall(wall_id, ocw_test["dataset"])
                candidate_group = random.choice(list(wall["groups"].values())[1:])
                # ...hasn't been sampled before for any wall
                if (
                    candidate_group["group_id"] not in seen_groups
                    # ...and doesn't contain any words already in the current wall
                    and all(word not in seen_words for word in candidate_group["gt_words"])
                ):
                    seen_walls.add(wall_id)
                    seen_groups.add(candidate_group["group_id"])
                    seen_words.update(candidate_group["gt_words"])
                    break

            filtered_test[i]["groups"][group_key] = candidate_group

        # Update macro statistics for each newly created wall
        filtered_test[i]["words"] = [
            word for group in filtered_test[i]["groups"].values() for word in group["gt_words"]
        ]
        random.shuffle(filtered_test[i]["words"])
        filtered_test[i]["gt_connections"] = [
            group["gt_connection"] for group in filtered_test[i]["groups"].values()
        ]
        filtered_test[i]["overall_human_performance"] = {
            "grouping": [
                group["human_performance"]["grouping"]
                for group in filtered_test[i]["groups"].values()
            ],
            "connections": [
                group["human_performance"]["connection"]
                for group in filtered_test[i]["groups"].values()
            ],
        }

    # Write the new dataset to a file
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Write original train and dev sets, unchanged
    with open(output_dir / TRAIN_FN, "w") as f:
        shutil.copyfile(path / TRAIN_FN, output_dir / TRAIN_FN)
    with open(output_dir / VALID_FN, "w") as f:
        shutil.copyfile(path / VALID_FN, output_dir / VALID_FN)
    # Write the new, modified test set
    with open(output_dir / TEST_FN, "w") as f:
        json.dump({"season_to_walls_map": season_to_walls_map, "dataset": filtered_test}, f)
    print(f":floppy_disk: Wrote modified dataset to '{output_dir.absolute()}'")


if __name__ == "__main__":
    typer.run(main)
