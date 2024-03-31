import tensorflow as tf
import scipy.spatial.distance as spd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score


def compute_mean_vector(feature):
    return np.mean(feature, axis=0)


def compute_distance(mean_feature, feature, distance_type='eucos'):
    if distance_type == 'eucos':
        query_distance = spd.euclidean(
            mean_feature, feature)/200. + spd.cosine(mean_feature, feature)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mean_feature, feature)/200.
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mean_feature, feature)
    else:
        print('Distance type unknown, valid distance type: eucos, euclidean, cosine')
    return query_distance


def compute_distance_dict(mean_feature, feature):
    eu_dist, cos_dist, eucos_dist = [], [], []
    for feat in feature:
        eu_dist += [spd.euclidean(mean_feature, feat)/200.]
        cos_dist += [spd.cosine(mean_feature, feat)]
        eucos_dist += [spd.euclidean(mean_feature, feat)/200. + spd.cosine(
            mean_feature, feat)]
    distances = {'eucos': eucos_dist, 'cosine': cos_dist, 'euclidean': eu_dist}
    return distances


def get_openmax_predict(openmax, threshold=0.0004):
    if openmax[10] >= threshold:
        res = 10
    else:
        temp = np.delete(openmax, -1)
        res = np.argmax(temp)
    return res


def convert_label(y):
    y[y != 10] = 0
    y[y == 10] = 1
    return y


def get_activations(x, model):
    partial_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer('logits').output,
            model.get_layer('softmax').output,
        ],
    )
    logits_output, softmax_output = partial_model.predict(x)
    return logits_output, softmax_output


def get_correct_classified(prob, y):
    pred_y = np.argmax(prob, axis=1)
    res = pred_y == y
    return res


def compute_roc(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    roc = {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auroc': auroc,
    }
    return roc


def compute_pr(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    pr = {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'aupr': aupr,
    }
    return pr


def plot_roc(roc):
    plt.plot(roc['fpr'], roc['tpr'])
    plt.grid('on')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUROC: {:.5f}'.format(roc['auroc']))


def plot_pr(pr):
    plt.plot(pr['recall'], pr['precision'])
    plt.grid('on')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUPR: {:.5f}'.format(pr['aupr']))


def image_show(img):
    plt.imshow(np.squeeze(img), cmap='gray')
    plt.show()
