from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
tqdm.pandas()


def get_res(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    minrp = np.minimum(precision, recall).max()
    roc_auc = roc_auc_score(y_true, y_pred)
    return [roc_auc, pr_auc, minrp]


class_weights = compute_class_weight(
    class_weight='balanced', classes=[0, 1], y=train_op)


def mortality_loss(y_true, y_pred):
    sample_weights = (1-y_true)*class_weights[0] + y_true*class_weights[1]
    bce = K.binary_crossentropy(y_true, y_pred)
    return K.mean(sample_weights*bce, axis=-1)


def forecast_loss(y_true, y_pred):
    return K.sum(y_true[:, V:]*(y_true[:, :V]-y_pred)**2, axis=-1)


def get_min_loss(weight):
    def min_loss(y_true, y_pred):
        return weight*y_pred
    return min_loss


class CustomCallback(Callback):
    def __init__(self, validation_data, batch_size):
        self.val_x, self.val_y = validation_data
        self.batch_size = batch_size
        super(Callback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(
            self.val_x, verbose=0, batch_size=self.batch_size)
        if type(y_pred) == type([]):
            y_pred = y_pred[0]
        precision, recall, thresholds = precision_recall_curve(
            self.val_y, y_pred)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(self.val_y, y_pred)
        logs['custom_metric'] = pr_auc + roc_auc
        print('val_aucs:', pr_auc, roc_auc)