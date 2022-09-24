# plotting utility for evaluation
import numpy as np
import matplotlib.pyplot as mplt
import matplotlib.ticker as ticker


def plot(x, y_original, y1, y2, y3, label1, label2, label3, outputFileLocation):
    # mplt.style.use("science")
    fig, ax = mplt.subplots(1, 3, figsize=(8,4))
    ax[0].plot(x, y1, "-", color="black", label=label1)
    # ax[0].set_title('$\sum$', usetex=True) # TODO: deploy latex
    ax[1].plot(x, y2, "-", color="black", label=label2)
    ax[1].set_title('todo')
    ax[2].plot(x, y3, "-", color="black", label=label3)
    ax[2].set_title('todo')

    for p in range(3):
        ax[p].plot(x, y_original, "--", color="black", label=label1)
        ax[p].set_xlabel('Epoch')
        ax[p].grid(True)
        # ax[p].legend()
        ax[p].xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax[p].set_ylim([0, 1])
    ax[0].set_ylabel('Accuracy')

    fig.tight_layout()
    mplt.savefig(outputFileLocation)


# entry point, run the test harness
if __name__ == '__main__':
    x = [i+1 for i in range(15)]
    y_original = [0.8054, 0.9403, 0.9523, 0.9572, 0.9604, 0.9628, 0.9646, 0.9662, 0.9676, 0.9687, 0.9696, 0.9707, 0.9716, 0.9717, 0.9728]
    y1 = [0.1460, 0.1191, 0.1299, 0.1438, 0.1396, 0.1332, 0.1296, 0.1303, 0.1310, 0.1319, 0.1330, 0.1332, 0.1331, 0.1332, 0.1328]  # todo: load data
    y2 = [0.8934, 0.9625, 0.9688, 0.9721, 0.9733, 0.9742, 0.9755, 0.9761, 0.9764, 0.9768, 0.9774, 0.9776, 0.9776, 0.9783, 0.9786]  # todo: load data
    y3 = [0.5565, 0.7620, 0.8619, 0.9078, 0.9315, 0.9462, 0.9517, 0.9558, 0.9605, 0.9630, 0.9645, 0.9656, 0.9667, 0.9674, 0.9683]  # todo: load data
    plot(x, y_original, y1, y2, y3, "p1", "p2", "p3", "plot.png")  # todo: get proper names
