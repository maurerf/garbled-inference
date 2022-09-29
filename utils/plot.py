# plotting utility for evaluation
import numpy as np
import matplotlib.pyplot as mplt
import matplotlib.ticker as ticker
from matplotlib import rc
import matplotlib.font_manager
from keras.datasets import mnist
from matplotlib import pyplot


# rc("text", usetex=False)


def plot(x, y_original, y1, y2, y3, label1, label2, label3, ylim, output_file_location):
    mplt.style.use(["science", "no-latex"])
    fig, ax = mplt.subplots(1, 3, figsize=(8, 3))
    ax[0].plot(x, y1, "-", color="black", label=label1)
    ax[0].set_title(label1)
    ax[1].plot(x, y2, "-", color="black", label=label2)
    ax[1].set_title(label2)
    ax[2].plot(x, y3, "-", color="black", label=label3)
    ax[2].set_title(label3)

    for p in range(3):
        ax[p].plot(x, y_original, "--", color="black", label=label1)
        ax[p].set_xlabel('Epoch')
        ax[p].grid(True)
        # ax[p].legend()
        ax[p].xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax[p].set_ylim(ylim)
    ax[0].set_ylabel('Accuracy')

    fig.tight_layout()
    mplt.savefig(output_file_location)


# entry point, run the test harness
if __name__ == '__main__':
    x = [i + 1 for i in range(15)]
    # RELU
    relu_y_orig = [0.8054, 0.9403, 0.9523, 0.9572, 0.9604, 0.9628, 0.9646, 0.9662, 0.9676, 0.9687, 0.9696, 0.9707,
                   0.9716, 0.9717, 0.9728]
    relu_y1 = [0.8490, 0.9492, 0.9599, 0.9652, 0.9671, 0.9701, 0.9724, 0.9736, 0.9750, 0.9762, 0.9768, 0.9773, 0.9778,
               0.9789, 0.9795]
    relu_y2 = [0.8934, 0.9625, 0.9688, 0.9721, 0.9733, 0.9742, 0.9755, 0.9761, 0.9764, 0.9768, 0.9774, 0.9776, 0.9776,
               0.9783, 0.9786]
    relu_y3 = [0.5565, 0.7620, 0.8619, 0.9078, 0.9315, 0.9462, 0.9517, 0.9558, 0.9605, 0.9630, 0.9645, 0.9656, 0.9667,
               0.9674, 0.9683]
    plot(x, relu_y_orig, relu_y1, relu_y2, relu_y3, "(a)", "(b)", "(c)", [0.65, 1.0], "benchmark-results/plot_relu.pdf")

    # TANH
    tanh_y_orig = [0.8330, 0.9377, 0.9517, 0.9590, 0.9637, 0.9664, 0.9677, 0.9698, 0.9707, 0.9716, 0.9733, 0.9734,
                   0.9741, 0.9740, 0.9755]
    tanh_y1 = [0.7515, 0.9156, 0.9318, 0.9404, 0.9459, 0.9504, 0.9538, 0.9560, 0.9579, 0.9598, 0.9614, 0.9629, 0.9638,
               0.9651, 0.9660]
    tanh_y2 = [0.8044, 0.9276, 0.9392, 0.9450, 0.9495, 0.9538, 0.9566, 0.9596, 0.9615, 0.9628, 0.9638, 0.9653, 0.9660,
               0.9671, 0.9679]
    tanh_y3 = [0.8085, 0.9222, 0.9355, 0.9432, 0.9490, 0.9527, 0.9553, 0.9571, 0.9593, 0.9605, 0.9613, 0.9624, 0.9631,
               0.9641, 0.9645]
    plot(x, tanh_y_orig, tanh_y1, tanh_y2, tanh_y3, "(a)", "(b)", "(c)", [0.65, 1.0], "benchmark-results/plot_tanh.pdf")

    # SWISH
    swish_y_orig = [0.8354, 0.9425, 0.9553, 0.9621, 0.9653, 0.9676, 0.9699, 0.9714, 0.9725, 0.9734, 0.9746, 0.9751,
                    0.9758, 0.9760, 0.9763]
    swish_y1 = [0.7224, 0.9331, 0.9510, 0.9596, 0.9642, 0.9668, 0.9693, 0.9711, 0.9720, 0.9736, 0.9743, 0.9755, 0.9758,
                0.9765, 0.9772]
    swish_y2 = [0.6942, 0.9289, 0.9487, 0.9575, 0.9619, 0.9658, 0.9684, 0.9698, 0.9712, 0.9730, 0.9736, 0.9742, 0.9754,
                0.9759, 0.9767]
    swish_y3 = [0.8077, 0.9417, 0.9550, 0.9610, 0.9652, 0.9679, 0.9697, 0.9708, 0.9720, 0.9728, 0.9736, 0.9739, 0.9748,
                0.9751, 0.9750]
    plot(x, swish_y_orig, swish_y1, swish_y2, swish_y3, "(a)", "(b)", "(c)", [0.65, 1.0], "benchmark-results"
                                                                                          "/plot_swish.pdf")

    # plot MNIST examples
    (train_X, train_y), _ = mnist.load_data()
    for i in range(3):
        pyplot.subplot(250 + 1 + i)
        pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
    pyplot.show()
