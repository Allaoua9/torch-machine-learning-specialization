import csv
import pickle
from collections import defaultdict

import pandas as pd
import torch
from numpy import genfromtxt, loadtxt


def normalizeRatings(Y, R):
    """
    Preprocess data by subtracting mean rating for every movie (every row).
    Only include real ratings R(i,j)=1.
    [Ynorm, Ymean] = normalizeRatings(Y, R) normalized Y so that each movie
    has a rating of 0 on average. Unrated moves then have a mean rating (0)
    Returns the mean rating in Ymean.
    """
    Ymean = ((Y * R).sum(dim=1) / (R.sum(dim=1))).reshape(-1, 1)
    Ynorm = Y - (Ymean * R)
    print(Ymean * R)
    return (Ynorm, Ymean)


def gen_user_vecs(user_vec, num_items):
    """given a user vector return:
    user predict maxtrix to match the size of item_vecs"""
    user_vecs = torch.tile(user_vec, (num_items, 1))
    return user_vecs


def load_precalc_params_small():
    file = open("./data/small_movies_X.csv", "rb")
    X = torch.from_numpy(loadtxt(file, delimiter=","))

    file = open("./data/small_movies_W.csv", "rb")
    W = torch.from_numpy(loadtxt(file, delimiter=","))

    file = open("./data/small_movies_b.csv", "rb")
    b = torch.from_numpy(loadtxt(file, delimiter=","))
    b = b.reshape(1, -1)
    num_movies, num_features = X.shape
    num_users, _ = W.shape
    return (X, W, b, num_movies, num_features, num_users)


def load_ratings_small():
    file = open("./data/small_movies_Y.csv", "rb")
    Y = torch.from_numpy(loadtxt(file, delimiter=","))

    file = open("./data/small_movies_R.csv", "rb")
    R = torch.from_numpy(loadtxt(file, delimiter=","))
    return (Y, R)


def load_Movie_List_pd():
    """returns df with and index of movies in the order they are in in the Y matrix"""
    df = pd.read_csv(
        "./data/small_movie_list.csv",
        header=0,
        index_col=0,
        delimiter=",",
        quotechar='"',
    )
    mlist = df["title"].to_list()
    return (mlist, df)


def load_data():
    """called to load preprepared data for the lab"""
    item_train = torch.from_numpy(
        genfromtxt("./data/content_item_train.csv", delimiter=",")
    )
    user_train = torch.from_numpy(
        genfromtxt("./data/content_user_train.csv", delimiter=",")
    )
    y_train = torch.from_numpy(genfromtxt("./data/content_y_train.csv", delimiter=","))
    with open(
        "./data/content_item_train_header.txt", newline=""
    ) as f:  # csv reader handles quoted strings better
        item_features = list(csv.reader(f))[0]
    with open("./data/content_user_train_header.txt", newline="") as f:
        user_features = list(csv.reader(f))[0]
    item_vecs = torch.from_numpy(
        genfromtxt("./data/content_item_vecs.csv", delimiter=",")
    )

    movie_dict = defaultdict(dict)
    count = 0
    #    with open('./data/movies.csv', newline='') as csvfile:
    with open("./data/content_movie_list.csv", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        for line in reader:
            if count == 0:
                count += 1  # skip header
                # print(line) print
            else:
                count += 1
                movie_id = int(line[0])
                movie_dict[movie_id]["title"] = line[1]
                movie_dict[movie_id]["genres"] = line[2]

    with open("./data/content_user_to_genre.pickle", "rb") as f:
        user_to_genre = pickle.load(f)

    return (
        item_train,
        user_train,
        y_train,
        item_features,
        user_features,
        item_vecs,
        movie_dict,
        user_to_genre,
    )


def format_predictions(sorted_ypu, sorted_items, movie_dict):
    predictions = []
    for i, (id, year, avg_rating, *_) in enumerate(sorted_items):
        predictions.append(
            [
                sorted_ypu[i][0].item(),
                int(id),
                avg_rating.item(),
                movie_dict[int(id)]["title"],
                movie_dict[int(id)]["genres"],
            ]
        )

    return pd.DataFrame(
        data=predictions,
        columns=["y_p", "movie_id", "avg_rating", "title", "genres"],
    )
