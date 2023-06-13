from utils import *
from prediction import ModelPrediction
from arguments import get_args
from adjustText import adjust_text
import seaborn as sns
from matplotlib.colors import ListedColormap


class PlotWall:
    def __init__(
        self,
        model_name,
        wall_id,
        contextual=False,
        shuffle_seed=0,
        seed=42,
        split="test",
        dim_reduction="tsne",
        dataset_path="./",
        save_path="./plots_new/",
    ):
        self.model_name = model_name
        self.wall_id = wall_id
        self.contextual = contextual
        self.shuffle_seed = shuffle_seed
        self.seed = seed
        self.split = split
        self.dim_reduction = dim_reduction
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.DATASET = load_hf_dataset(self.dataset_path)

    def plot(self):
        for i in self.DATASET[self.split]:
            if i["wall_id"] == self.wall_id:
                wall = i
                break
        spare_model, clf = ModelPrediction(
            model_name=self.model_name,
            contextual=self.contextual,
            seed=self.seed,
            dataset_path=self.dataset_path,
        ).load_model()
        contextual = "contextual" if self.contextual else "static"
        saved_file = (
            "model_"
            + self.model_name.replace("/", "-")
            + "_"
            + contextual
            + "_wall_"
            + self.wall_id
            + "_"
            + self.dim_reduction
            + "_shuffleSeed_"
            + str(self.shuffle_seed)
            + ".pdf"
        )

        # get embedding of the wall
        if isinstance(self.shuffle_seed, int):
            wall["words"] = random.Random(self.shuffle_seed).sample(
                wall["words"], len(wall["words"])
            )
        # step 1 => get model's embeddings
        if self.contextual or self.model_name == "elmo":
            wall_embed = get_embeddings(spare_model, wall["words"])
            if self.model_name == "elmo" and not self.contextual:
                # first 1024 embeddings of elmo are static
                wall_embed = wall_embed[:, :1024]
        else:
            wall_embed = get_embeddings_static(spare_model, wall["words"])
        clf_embeds = clf.fit_predict(wall_embed.detach().cpu())

        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        if self.dim_reduction == "tsne":
            reduction = load_tsne(seed=0).fit_transform(wall_embed.detach().cpu())
        elif self.dim_reduction == "pca":
            reduction = load_pca(seed=self.seed).fit_transform(wall_embed.detach().cpu())
        elif self.dim_reduction == "kernel_pca":
            reduction = load_kpca(seed=self.seed).fit_transform(wall_embed.detach().cpu())
        else:
            raise ValueError("No Valid Dimentionality Reduction Was Found")

        # label = load_clf().fit_predict(reduction)
        label = clf_embeds
        # Getting unique labels
        wall1_default = (
            wall["groups"]["group_1"]["gt_words"]
            + wall["groups"]["group_2"]["gt_words"]
            + wall["groups"]["group_3"]["gt_words"]
            + wall["groups"]["group_4"]["gt_words"]
        )
        connections = wall["gt_connections"]
        emmbedes_lst = wall["words"]
        LABEL_TRUE = clue2group(emmbedes_lst, wall1_default)
        U_LABEL_TRUE = np.unique(LABEL_TRUE)
        # centroids = clf.cluster_centers_
        # colors = ['purple', 'green', 'red', 'blue']
        palette = ["#648FFF", "#785EF0", "#DC267F", "#FE6100"]
        my_cmap = ListedColormap(sns.color_palette(palette).as_hex())

        font = {"family": "Times New Roman", "size": 10}

        plt.rc("font", **font)
        plt.figure(figsize=(4, 4))
        # plotting the results:
        # draw enclosure
        for i in label:
            points = reduction[label == i]
            # get convex hull
            hull = ConvexHull(points)
            # get x and y coordinates
            # repeat last point to close the polygon
            x_hull = np.append(points[hull.vertices, 0], points[hull.vertices, 0][0])
            y_hull = np.append(points[hull.vertices, 1], points[hull.vertices, 1][0])
            # # plot shape
            # plt.fill(x_hull, y_hull, alpha=0.3, c='gainsboro')

            # interpolate
            dist = np.sqrt((x_hull[:-1] - x_hull[1:]) ** 2 + (y_hull[:-1] - y_hull[1:]) ** 2)
            dist_along = np.concatenate(([0], dist.cumsum()))
            spline, u = interpolate.splprep([x_hull, y_hull], u=dist_along, s=0)
            interp_d = np.linspace(dist_along[0], dist_along[-1], 100)
            interp_x, interp_y = interpolate.splev(interp_d, spline)
            # plot shape
            plt.fill(interp_x, interp_y, "--", c="gainsboro", alpha=0.1)
        markers = ["o", "s", "v", "D"]
        for i in U_LABEL_TRUE:
            plt.scatter(
                reduction[LABEL_TRUE == i, 0],
                reduction[LABEL_TRUE == i, 1],
                label="Group " + str(i + 1) + ": " + connections[i],
                s=50,
                color=my_cmap(i),
                marker=markers[i],
                alpha=1,
            )
        # plt.annotate(name, xy=(x, y), xytext=(0, 7), textcoords='offset points', ha='center', va='center')
        texts = [
            plt.text(x, y, name, fontsize=font["size"])
            for name, x, y in zip(emmbedes_lst, reduction[:, 0], reduction[:, 1])
        ]

        # Plot the centroids as a black X
        # plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='k')
        adjust_text(texts)
        # plt.grid(b=None)
        plt.legend(loc="best", fontsize=font["size"] - 2.5)
        ax = plt.gca()
        leg = ax.get_legend()
        leg.legendHandles[0].set_color(my_cmap(0))
        leg.legendHandles[1].set_color(my_cmap(1))
        leg.legendHandles[2].set_color(my_cmap(2))
        leg.legendHandles[3].set_color(my_cmap(3))
        #

        # plt.show()
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        plt.savefig(self.save_path + saved_file, dpi=500, facecolor="white", bbox_inches="tight")


if __name__ == "__main__":
    args = get_args()
    PlotWall(
        model_name=args.model_name,
        wall_id=args.plot,
        contextual=args.contextual,
        shuffle_seed=args.shuffle_seed,
        seed=args.seed,
        split=args.split,
        dim_reduction=args.dim_reduction,
        dataset_path=args.dataset_path,
    ).plot()
