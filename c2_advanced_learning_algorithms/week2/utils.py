import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt

dkcolors = plt.cm.Paired((1, 3, 7, 9, 5, 11))
dkcolors_map = mpl.colors.ListedColormap(dkcolors)


def plt_mc_data(
    ax,
    X,
    y,
    classes,
    class_labels=None,
    map=plt.cm.Paired,
    legend=False,
    size=50,
    m="o",
    equal_xy=False,
):
    """Plot multiclass data. Note, if equal_xy is True, setting ylim on the plot may not work"""
    for i in range(classes):
        idx = np.where(y == i)
        col = len(idx[0]) * [i]
        label = class_labels[i] if class_labels else "c{}".format(i)
        # this didn't work on coursera but did in local version
        # ax.scatter(X[idx, 0], X[idx, 1],  marker=m,
        #            c=col, vmin=0, vmax=map.N, cmap=map,
        #            s=size, label=label)
        ax.scatter(
            X[idx, 0],
            X[idx, 1],
            marker=m,
            color=map(col),
            vmin=0,
            vmax=map.N,
            s=size,
            label=label,
        )
    if legend:
        ax.legend()
    if equal_xy:
        ax.axis("equal")


def plt_mc(X_train, y_train, classes):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    plt_mc_data(
        ax,
        X_train,
        y_train,
        classes,
        map=dkcolors_map,
        legend=True,
        size=50,
        equal_xy=False,
    )
    ax.set_title("Multiclass Data")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    # for c in css:
    #    circ = plt.Circle(centers[c], 2*std, color=dkcolors_map(c), clip_on=False, fill=False, lw=0.5)
    #    ax.add_patch(circ)
    plt.show()


def plot_cat_decision_boundary_mc(ax, X, predict, vector=True):
    # create a mesh to points to plot
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = max(x_max - x_min, y_max - y_min) / 100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    points = np.c_[xx.ravel(), yy.ravel()]
    # print("points", points.shape)
    # print("xx.shape", xx.shape)

    # make predictions for each point in mesh
    if vector:
        Z = predict(points)
    else:
        Z = np.zeros((len(points),))
        for i in range(len(points)):
            Z[i] = predict(points[i].reshape(1, 2))
    Z = Z.reshape(xx.shape)

    # contour plot highlights boundaries between values - classes in this case
    ax.contour(xx, yy, Z, linewidths=1)


def plt_cat_mc(X_train, y_train, model, classes):
    # make a model for plotting routines to call
    model_predict = lambda Xl: torch.argmax(model(torch.from_numpy(Xl)), axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

    # add the original data to the decison boundary
    plt_mc_data(ax, X_train, y_train, classes, map=dkcolors_map, legend=True)
    # plot the decison boundary.
    plot_cat_decision_boundary_mc(ax, X_train, model_predict, vector=True)
    ax.set_title("model decision boundary")

    plt.xlabel(r"$x_0$")
    plt.ylabel(r"$x_1$")
    plt.show()


def widgvis(fig):
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False


def plot_loss_tf(history):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    widgvis(fig)
    ax.plot(history, label="loss")
    ax.set_ylim([0, 2])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("loss (cost)")
    ax.legend()
    ax.grid(True)
    plt.show()


def display_digit(X):
    """display a single digit. The input is one digit (400,)."""
    fig, ax = plt.subplots(1, 1, figsize=(0.5, 0.5))
    widgvis(fig)
    X_reshaped = X.reshape((20, 20)).T
    # Display the image
    ax.imshow(X_reshaped, cmap="gray")
    plt.show()


def load_data():
    X = torch.from_numpy(np.load("data/X.npy"))
    y = torch.from_numpy(np.load("data/y.npy"))

    return X, y
