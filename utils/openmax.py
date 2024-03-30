import numpy as np
import pickle

from utils.evt_fitting import query_weibull
from utils.evt_fitting import weibull_tailfitting
from utils.openmax_utils import *

IMG_DIM = 28
NCLASSES = 10
ALPHA_RANK = 1
WEIBULL_TAIL_SIZE = 10
MODEL_PATH = 'models/weibull_model.pkl'

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def create_model(model, data):
    x_train, x_test, y_train, y_test = data
    x_all = np.concatenate((x_train, x_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)

    logits_output, softmax_output = get_activations(x_all, model)
    correct_index = get_correct_classified(softmax_output, y_all)

    logits_correct = logits_output[correct_index]
    y_correct = y_all[correct_index]

    av_map = {}

    for label in labels:
        av_map[label] = logits_correct[(y_correct == label), :]

    feature_mean = []
    feature_distance = []

    for label in labels:
        mean = compute_mean_vector(av_map[label])
        distance = compute_distance_dict(mean, av_map[label])
        feature_mean.append(mean)
        feature_distance.append(distance)

    build_weibull(mean=feature_mean, distance=feature_distance,
                  tail=WEIBULL_TAIL_SIZE)


def build_weibull(mean, distance, tail):
    weibull_model = {}

    for label in labels:
        weibull_model[label] = {}
        weibull = weibull_tailfitting(
            mean[label], distance[label], tailsize=tail)
        weibull_model[label] = weibull

    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(weibull_model, file)


def recalibrate_scores(weibull_model, labels, activation_vector, alpharank=ALPHA_RANK, distance_type='eucos'):
    ranked_list = activation_vector.argsort().ravel()[::-1]
    alpha_weights = [
        ((alpharank+1) - i) / float(alpharank) for i in range(1, alpharank+1)
    ]
    ranked_alpha = np.zeros(NCLASSES)

    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    # print(ranked_alpha)

    openmax_scores = []
    openmax_scores_u = []

    for label in labels:
        weibull = query_weibull(label, weibull_model, distance_type)
        av_distance = compute_distance(
            weibull[1], activation_vector.ravel())
        wscore = weibull[2][0].w_score(av_distance)
        # print(f'wscore_{label}: {wscore}')
        modified_score = activation_vector[0][label] * \
            (1 - wscore*ranked_alpha[label])
        openmax_scores += [modified_score]
        openmax_scores_u += [activation_vector[0][label] - modified_score]

    openmax_scores = np.array(openmax_scores)
    openmax_scores_u = np.array(openmax_scores_u)

    # print(f'activation_vector: {activation_vector}')
    # print(f'openmax_scores: {openmax_scores}')
    # print(f'openmax_scores_u: {np.sum(openmax_scores_u)}')

    openmax_probab, prob_u = computeOpenMaxProbability(
        openmax_scores, openmax_scores_u)
    return openmax_probab, prob_u


def computeOpenMaxProbability(openmax_scores, openmax_scores_u):
    e_k = np.exp(openmax_scores)
    e_u = np.exp(np.sum(openmax_scores_u))
    openmax_arr = np.concatenate((e_k, e_u), axis=None)
    total_denominator = np.sum(openmax_arr)
    prob_k = e_k / total_denominator
    prob_u = e_u / total_denominator
    res = np.concatenate((prob_k, prob_u), axis=None)
    return res, prob_u


def compute_openmax(activation_vector):
    with open(MODEL_PATH, 'rb') as file:
        weibull_model = pickle.load(file)
    openmax, prob_u = recalibrate_scores(
        weibull_model, labels, activation_vector)
    return openmax, prob_u
