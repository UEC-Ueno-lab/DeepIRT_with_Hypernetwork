import logging
import numpy as np
import tensorflow as tf
from sklearn import metrics
from utils import getLogger
from utils import ProgressBar
# Code reused from https://github.com/ckyeungac/DeepIRT.git


def compute_auc(all_label, all_pred):
    return metrics.roc_auc_score(all_label, all_pred)


def compute_accuracy(all_label, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_label, all_pred)


def compute_diff_score(all_target, all_pred):
    diff = np.abs(all_target - all_pred)
    return np.mean(diff)


def compute_precision(all_target, all_pred):
    return metrics.precision_score(all_target, all_pred, average=None)


def compute_recall(all_target, all_pred):
    return metrics.recall_score(all_target, all_pred, average=None)


def compute_f1_score(all_target, all_pred):
    return metrics.f1_score(all_target, all_pred, average=None)


def binaryEntropy(label, pred, mod="avg"):
    loss = label * np.log(np.maximum(1e-10, pred)) + \
           (1.0 - label) * np.log(np.maximum(1e-10, 1.0 - pred))
    if mod == 'avg':
        return np.average(loss) * (-1.0)
    elif mod == 'sum':
        return - loss.sum()
    else:
        assert False


def run_model(model, args, s_data, q_data, qa_data, mode):
    """
    Run one epoch.

    Parameters:
        - q_data: Shape (num_train_samples, seq_len)
        - qa_data: Shape (num_train_samples, seq_len)
    """
    shuffle_index = np.random.permutation(q_data.shape[0])
    if mode == 'train':
        s_data_shuffled = s_data[shuffle_index]
        q_data_shuffled = q_data[shuffle_index]
        qa_data_shuffled = qa_data[shuffle_index]
    # don't shuffle the data when validate and test the model
    else:
        s_data_shuffled = s_data
        q_data_shuffled = q_data
        qa_data_shuffled = qa_data

    training_step = q_data.shape[0] // args.batch_size
    if args.show:
        bar = ProgressBar(mode, max=training_step)

    pred_list = list()
    label_list = list()
    all_student_abilities_list = list()
    all_difficulties_list = list()
    all_pred_list = list()
    all_skill_difficulties_list = list()
    for step in range(training_step):
        if args.show:
            bar.next()

        s_data_batch = s_data_shuffled[step * args.batch_size:(step + 1) * args.batch_size, :]
        q_data_batch = q_data_shuffled[step * args.batch_size:(step + 1) * args.batch_size, :]
        qa_data_batch = qa_data_shuffled[step * args.batch_size:(step + 1) * args.batch_size, :]

        # qa : exercise index + answer(0 or 1)*skill_number
        label = qa_data_batch[:, :]
        label = label.astype(np.int)
        label_batch = (label - 1) // args.n_skills  # convert to {-1, 0, 1}
        label_batch = label_batch.astype(np.float)

        feed_dict = {
            model.s_data: s_data_batch,
            model.q_data: q_data_batch,
            model.qa_data: qa_data_batch,
            model.label: label_batch
        }
        if mode == "train":
            pred_, _, student_abilities_, question_difficulties_, skill_difficulties_, pred_list_ = model.sess.run(
                [model.pred, model.train_op, model.student_abilities, model.question_difficulties,
                 model.skill_difficulties, model.pred_value_list], feed_dict=feed_dict)
        else:
            pred_, student_abilities_, question_difficulties_, skill_difficulties_, pred_list_ = model.sess.run(
                [model.pred, model.student_abilities, model.question_difficulties, model.skill_difficulties,
                 model.pred_value_list], feed_dict=feed_dict)

        label_flat = np.asarray(label_batch).reshape((-1))
        pred_flat = np.asarray(pred_).reshape((-1))
        index_flat = np.flatnonzero(label_flat != -1.).tolist()
        all_student_abilities_list.append(student_abilities_)
        all_pred_list.append(pred_list_)
        all_difficulties_list.append(question_difficulties_)
        all_skill_difficulties_list.append(skill_difficulties_)

        label_list.append(label_flat[index_flat])
        pred_list.append(pred_flat[index_flat])
    if args.show:
        bar.finish()

    all_label = np.concatenate(label_list, axis=0)
    all_pred = np.concatenate(pred_list, axis=0)

    auc = compute_auc(all_label, all_pred)
    Diff = compute_diff_score(all_label, all_pred)
    accuracy = compute_accuracy(all_label, all_pred)
    loss = binaryEntropy(all_label, all_pred)
    precision = compute_precision(all_label, all_pred)
    recall = compute_recall(all_label, all_pred)
    f1_score = compute_f1_score(all_label, all_pred)

    return loss, accuracy, auc, f1_score, student_abilities_, question_difficulties_, skill_difficulties_, Diff, \
           precision, recall, label_batch, all_pred, all_student_abilities_list, all_pred_list, all_difficulties_list, \
           all_skill_difficulties_list
