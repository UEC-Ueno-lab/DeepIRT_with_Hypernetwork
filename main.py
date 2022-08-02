import argparse
import datetime
import logging
import csv
import numpy as np
import tensorflow as tf
import os
from load_data import DataLoader
from model import DeepIRTModel
from model_skill import DeepIRTModel_skill
from run import run_model
from run_skill import run_model_skill
from utils import getLogger
from configs import ModelConfigFactory
# Code reused from https://github.com/ckyeungac/DeepIRT.git
# set logger
logger = getLogger('Deep-IRT-model-HN')

# argument parser
parser = argparse.ArgumentParser()

# dataset can be assist2009_akt, assist2015_akt, assist2017_akt, statics2011_akt, assist2017, assist2009, junyi, Eedi
parser.add_argument('--dataset', default='assist2009_akt', )
parser.add_argument('--mode', type=str, default='both')
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--cpu', type=bool, default=False)
parser.add_argument('--n_epochs', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--train', type=bool, default=None)
parser.add_argument('--show', type=bool, default=None)
parser.add_argument('--learning_rate', type=float, default=None)
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--use_ogive_model', type=bool, default=False)

# parameter for the dataset
parser.add_argument('--seq_len', type=int, default=None)
parser.add_argument('--n_questions', type=int, default=None)
parser.add_argument('--n_skills', type=int, default=None)
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--data_name', type=str, default=None)

# parameter for the DKVMN model
parser.add_argument('--memory_size', type=int, default=None)
parser.add_argument('--key_memory_state_dim', type=int, default=None)
parser.add_argument('--value_memory_state_dim', type=int, default=None)
parser.add_argument('--summary_vector_output_dim', type=int, default=None)

# parameter for hyper-network and number of pattern
parser.add_argument('--delta_1', type=float, default=None)
parser.add_argument('--delta_2', type=float, default=None)
parser.add_argument('--rounds', type=int, default=None)
parser.add_argument('--num_pattern', type=int, default=None)

_args = parser.parse_args()
args = ModelConfigFactory.create_model_config(_args)
logger.info("Model Config: {}".format(args))

# create directory
for directory in [args.checkpoint_dir, args.result_log_dir, args.tensorboard_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)


def train(model, train_q_data, train_s_data, train_qa_data, valid_q_data, valid_s_data, valid_qa_data, test_q_data,
          test_s_data, test_qa_data, result_log_path, args):
    best_loss = 1e6
    best_acc = 0.0
    best_auc = 0.0
    best_epoch = 0.0
    train_aucs = []
    train_losses = []
    # write the results log
    with open(result_log_path, 'w') as f:
        result_msg = "{},{},{},{},{},{},{}\n".format(
            'epoch',
            'train_auc', 'train_accuracy', 'train_loss',
            'valid_auc', 'valid_accuracy', 'valid_loss'
        )
        f.write(result_msg)
    for epoch in range(args.n_epochs):
        train_loss, train_accuracy, train_auc, train_f1_score, train_ability, train_difficult, train_skill_difficult, train_Diff, train_precision, train_recall, train_label_batch, train_all_pred, all_ability_train, pred_list_train, all_difficulties_train, all_skill_difficulties_train = run_model(
            model, args, train_s_data, train_q_data, train_qa_data, mode='train'
        )
        valid_loss, valid_accuracy, valid_auc, valid_f1_score, valid_ability, valid_difficult, valid_skill_difficult, valid_Diff, valid_precision, valid_recall, valid_label_batch, valid_all_pred, all_ability_valid, pred_list_valid, all_difficulties_valid, all_skill_difficulties_valid = run_model(
            model, args, valid_s_data, valid_q_data, valid_qa_data, mode='valid'
        )
        test_loss, test_accuracy, test_auc, test_f1_score, test_ability, test_difficult, test_skill_difficult, test_Diff, test_precision, test_recall, test_label_batch, test_all_pred, all_ability_test, pred_list_test, all_difficulties_test, all_skill_difficulties_test = run_model(
            model, args, test_s_data, test_q_data, test_qa_data, mode='valid'
        )

        # add to log
        msg = "\n[Epoch {}/{}] Training result:      AUC: {:.2f}%\t Acc: {:.2f}%\t Loss: {:.4f}%\t F1: {:.2f}　{:.2f}".format(
            epoch + 1, args.n_epochs, train_auc * 100, train_accuracy * 100, train_loss, train_f1_score[0] * 100,
            train_f1_score[1] * 100
        )
        msg += "\n[Epoch {}/{}] Validation result:    AUC: {:.2f}%\t Acc: {:.2f}%\t Loss: {:.4f}%\t F1: {:.2f}　{:.2f}".format(
            epoch + 1, args.n_epochs, valid_auc * 100, valid_accuracy * 100, valid_loss, valid_f1_score[0] * 100,
            valid_f1_score[1] * 100
        )
        msg += "\n[Epoch {}/{}] Test result:    AUC: {:.2f}%\t Acc: {:.2f}%\t Loss: {:.4f}".format(
            epoch + 1, args.n_epochs, test_auc * 100, test_accuracy * 100, test_loss, test_f1_score[0] * 100,
            test_f1_score[1] * 100
        )
        logger.info(msg)

        # write epoch result
        with open(result_log_path, 'a') as f:
            result_msg = "{},{},{},{},{},{},{}\n".format(
                epoch,
                train_auc, train_accuracy, train_loss,
                valid_auc, valid_accuracy, valid_loss,
                test_auc, test_accuracy, test_loss
            )
            f.write(result_msg)

        # add to tensorboard
        tf_summary = tf.Summary(
            value=[
                tf.Summary.Value(tag="train_loss", simple_value=train_loss),
                tf.Summary.Value(tag="train_auc", simple_value=train_auc),
                tf.Summary.Value(tag="train_accuracy", simple_value=train_accuracy),
                tf.Summary.Value(tag="valid_loss", simple_value=valid_loss),
                tf.Summary.Value(tag="valid_auc", simple_value=valid_auc),
                tf.Summary.Value(tag="valid_accuracy", simple_value=valid_accuracy),
            ]
        )
        model.tensorboard_writer.add_summary(tf_summary, epoch)
        if epoch == 1:
            best_auc = valid_auc
        # save the model if the loss is lower
        if valid_auc > best_auc:
            best_loss = valid_loss
            best_acc = valid_accuracy
            best_auc = valid_auc
            best_pre = valid_precision
            best_rec = valid_recall
            best_f1_score = valid_f1_score
            best_epoch = epoch + 1
            ability = valid_ability
            difficult = valid_difficult
            all_ability = all_ability_valid
            all_pred = pred_list_valid
            all_difficulties = all_difficulties_valid
            skill_difficult = valid_skill_difficult
            all_skill_difficulties = all_skill_difficulties_valid

            if args.save:
                model_dir = "ep{:03d}-auc{:.0f}-acc{:.0f}".format(
                    epoch + 1, valid_auc * 100, valid_accuracy * 100,
                )
                model_name = "Deep-IRT-HN"
                save_path = os.path.join(args.checkpoint_dir, model_dir, model_name, '.ckpt')

                logger.info("Model improved. Save model to {}".format(save_path))
            else:
                logger.info("Model improved.")

        if epoch - best_epoch > 40:
            break

    # print out the final result
    msg = "Best result at epoch {}: AUC: {:.2f}%\t Accuracy: {:.2f}%\t Loss: {:.4f}\t F1: {:.2f}///{:.2f}".format(
        best_epoch, best_auc * 100, best_acc * 100, best_loss, best_f1_score[0] * 100, best_f1_score[1] * 100
    )
    logger.info(msg)
    return best_auc, best_acc, best_loss, best_f1_score[0], best_f1_score[1], ability, difficult, skill_difficult, \
           best_pre[0], best_pre[1], best_rec[0], best_rec[
               1], all_ability, all_pred, all_difficulties, all_skill_difficulties


def train_skill(model, train_q_data, train_qa_data, valid_q_data, valid_qa_data, test_q_data, test_qa_data,
                result_log_path, args):
    saver = tf.train.Saver()
    best_loss = 1e6
    best_acc = 0.0
    best_auc = 0.0
    best_epoch = 0.0

    with open(result_log_path, 'w') as f:
        result_msg = "{},{},{},{},{},{},{}\n".format(
            'epoch',
            'train_auc', 'train_accuracy', 'train_loss',
            'valid_auc', 'valid_accuracy', 'valid_loss'
        )
        f.write(result_msg)
    for epoch in range(args.n_epochs):

        train_loss, train_accuracy, train_auc, train_f1_score, train_ability, train_difficult, train_Diff, train_precision, train_recall, train_label_batch, train_all_pred, all_ability_train, pred_list_train, all_difficulties_train = run_model_skill(
            model, args, train_q_data, train_qa_data, mode='train'
        )
        valid_loss, valid_accuracy, valid_auc, valid_f1_score, valid_ability, valid_difficult, valid_Diff, valid_precision, valid_recall, valid_label_batch, valid_all_pred, all_ability_valid, pred_list_valid, all_difficulties_valid = run_model_skill(
            model, args, valid_q_data, valid_qa_data, mode='valid'
        )
        test_loss, test_accuracy, test_auc, test_f1_score, test_ability, test_difficult, test_Diff, test_precision, test_recall, test_label_batch, test_all_pred, all_ability_test, pred_list_test, all_difficulties_test = run_model_skill(
            model, args, test_q_data, test_qa_data, mode='valid'
        )
        # add to log
        msg = "\n[Epoch {}/{}] Training result:      AUC: {:.2f}%\t Acc: {:.2f}%\t Loss: {:.4f}%\t F1: {:.2f}　{:.2f}".format(
            epoch + 1, args.n_epochs, train_auc * 100, train_accuracy * 100, train_loss, train_f1_score[0] * 100,
            train_f1_score[1] * 100
        )
        msg += "\n[Epoch {}/{}] Validation result:    AUC: {:.2f}%\t Acc: {:.2f}%\t Loss: {:.4f}%\t F1: {:.2f}　{:.2f}".format(
            epoch + 1, args.n_epochs, valid_auc * 100, valid_accuracy * 100, valid_loss, valid_f1_score[0] * 100,
            valid_f1_score[1] * 100
        )
        msg += "\n[Epoch {}/{}] Test result:    AUC: {:.2f}%\t Acc: {:.2f}%\t Loss: {:.4f}".format(
            epoch + 1, args.n_epochs, test_auc * 100, test_accuracy * 100, test_loss, test_f1_score[0] * 100,
            test_f1_score[1] * 100
        )
        logger.info(msg)

        # write epoch result
        with open(result_log_path, 'a') as f:
            result_msg = "{},{},{},{},{},{},{}\n".format(
                epoch,
                train_auc, train_accuracy, train_loss,
                valid_auc, valid_accuracy, valid_loss
            )
            f.write(result_msg)

        # add to tensorboard
        tf_summary = tf.Summary(
            value=[
                tf.Summary.Value(tag="train_loss", simple_value=train_loss),
                tf.Summary.Value(tag="train_auc", simple_value=train_auc),
                tf.Summary.Value(tag="train_accuracy", simple_value=train_accuracy),
                tf.Summary.Value(tag="valid_loss", simple_value=valid_loss),
                tf.Summary.Value(tag="valid_auc", simple_value=valid_auc),
                tf.Summary.Value(tag="valid_accuracy", simple_value=valid_accuracy),
            ]
        )
        model.tensorboard_writer.add_summary(tf_summary, epoch)

        # save the model if the loss is lower
        if valid_auc > best_auc:
            best_loss = valid_loss
            best_acc = valid_accuracy
            best_auc = valid_auc
            best_pre = valid_precision
            best_rec = valid_recall
            best_f1_score = valid_f1_score
            best_epoch = epoch + 1
            ability = valid_ability
            difficult = valid_difficult
            all_ability = all_ability_valid
            all_pred = pred_list_valid
            all_difficulties = all_difficulties_valid

            if args.save:
                model_dir = "ep{:03d}-auc{:.0f}-acc{:.0f}".format(
                    epoch + 1, valid_auc * 100, valid_accuracy * 100,
                )
                model_name = "Deep-IRT-HN"
                save_path = os.path.join(args.checkpoint_dir, model_dir, model_name)
                saver.save(sess=model.sess, save_path=save_path)
                logger.info("Model improved. Save model to {}".format(save_path))
            else:
                logger.info("Model improved.")
        if epoch - best_epoch > 40:
            break

    # print out the final result
    msg = "Best result at epoch {}: AUC: {:.2f}\t Accuracy: {:.2f}\t Loss: {:.4f}\t F1: {:.2f}///{:.2f}".format(
        best_epoch, best_auc * 100, best_acc * 100, best_loss, best_f1_score[0] * 100, best_f1_score[1] * 100
    )
    logger.info(msg)
    return best_auc, best_acc, best_loss, best_f1_score[0], best_f1_score[1], ability, difficult, best_pre[0], best_pre[
        1], best_rec[0], best_rec[1], all_ability, all_pred, all_difficulties


def cross_validation():
    tf.set_random_seed(1234)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    aucs, accs, losses, f1_scores1, f1_scores2, pre1s, pre2s, rec1s, rec2s = list(), list(), list(), list(), list(), list(), list(), list(), list()
    for i in range(5):
        tf.reset_default_graph()
        logger.info("Cross Validation {}".format(i + 1))
        result_csv_path = os.path.join(args.result_log_dir, 'fold-{}-result'.format(i + 1) + '.csv')

        with tf.Session(config=config) as sess:
            data_loader = DataLoader(args.n_questions, args.n_skills, args.seq_len, ',')
            if args.data_name in ['assist2009_updated', 'assist2017', 'assist2015', 'statics', 'junyi']:
                model = DeepIRTModel_skill(args, sess, name="Deep-IRT_skill")
            else:
                model = DeepIRTModel(args, sess, name="Deep-IRT")
            sess.run(tf.global_variables_initializer())
            if args.train:
                train_data_path = os.path.join(args.data_dir, args.data_name + '_train{}.csv'.format(i + 1))
                valid_data_path = os.path.join(args.data_dir, args.data_name + '_valid{}.csv'.format(i + 1))
                test_data_path = os.path.join(args.data_dir, args.data_name + '_test{}.csv'.format(i + 1))
                logger.info("Reading {} and {}".format(train_data_path, valid_data_path))

                # the datasets with both item and skill
                if args.data_name in ['assist2009_updated', 'assist2017', 'assist2015', 'statics', 'junyi']:
                    train_q_data, train_qa_data = data_loader.load_data_skill(train_data_path)
                    valid_q_data, valid_qa_data = data_loader.load_data_skill(valid_data_path)
                    test_q_data, test_qa_data = data_loader.load_data_skill(test_data_path)

                    auc, acc, loss, f1_score1, f1_score2, ability, difficult, pre1, pre2, rec1, rec2, all_ability, all_pred, all_difficulties = train_skill(
                        model,
                        train_q_data, train_qa_data,
                        valid_q_data, valid_qa_data,
                        test_q_data, test_qa_data,
                        result_log_path=result_csv_path,
                        args=args
                    )
                    all_skill_difficulties = all_difficulties
                else:
                    train_q_data, train_s_data, train_qa_data = data_loader.load_data(train_data_path, mode='train')
                    valid_q_data, valid_s_data, valid_qa_data = data_loader.load_data(valid_data_path, mode='valid')
                    test_q_data, test_s_data, test_qa_data = data_loader.load_data(test_data_path, mode='valid')

                    auc, acc, loss, f1_score1, f1_score2, ability, difficult, skill_difficult, pre1, pre2, rec1, rec2, all_ability, all_pred, all_difficulties, all_skill_difficulties = train(
                        model,
                        train_q_data, train_s_data, train_qa_data,
                        valid_q_data, valid_s_data, valid_qa_data,
                        test_q_data, test_s_data, test_qa_data,
                        result_log_path=result_csv_path,
                        args=args
                    )

                all_ability = np.asarray(all_ability).reshape([-1, args.seq_len])
                all_pred = np.asarray(all_pred).reshape([-1, args.seq_len])
                all_difficulties = np.asarray(all_difficulties).reshape([-1, args.seq_len])
                all_skill_difficulties = np.asarray(all_skill_difficulties).reshape([-1, args.seq_len])
                ability_csv_path = os.path.join(args.result_log_dir, '{}-ability'.format(args.data_name) + '.csv')
                pred_csv_path = os.path.join(args.result_log_dir, '{}-pred'.format(args.data_name) + '.csv')
                difficulty_csv_path = os.path.join(args.result_log_dir, '{}-difficulty'.format(args.data_name) + '.csv')
                skill_difficulty_csv_path = os.path.join(args.result_log_dir, '{}-skill_difficulty'.format(args.data_name) + '.csv')

                # write the explanatory parameters to files
                with open(ability_csv_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerows(all_ability)
                with open(pred_csv_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerows(all_pred)
                with open(difficulty_csv_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerows(all_difficulties)
                with open(skill_difficulty_csv_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerows(all_skill_difficulties)
                aucs.append(auc)
                accs.append(acc)
                losses.append(loss)
                f1_scores1.append(f1_score1)
                f1_scores2.append(f1_score2)
                pre1s.append(pre1)
                pre2s.append(pre2)
                rec1s.append(rec1)
                rec2s.append(rec2)
    # print the best result
    cross_validation_msg = "Cross Validation Result:\n"
    cross_validation_msg += "AUC: {:.2f} +/- {:.2f}\n".format(np.average(aucs) * 100, np.std(aucs) * 100)
    cross_validation_msg += "Accuracy: {:.2f} +/- {:.2f}\n".format(np.average(accs) * 100, np.std(accs) * 100)
    cross_validation_msg += "Loss: {:.2f} +/- {:.2f}\n".format(np.average(losses), np.std(losses))
    cross_validation_msg += "f1_score: {:.2f} +/- {:.2f} /// {:.2f} +/- {:.2f}\n".format(np.average(f1_scores1) * 100,
                                                                                         np.std(f1_scores1) * 100,
                                                                                         np.average(f1_scores2) * 100,
                                                                                         np.std(f1_scores2) * 100)
    logger.info(cross_validation_msg)

    # write result
    result_msg = datetime.datetime.now().strftime("%Y-%m-%dT%H%M") + ','
    result_msg += str(args.dataset) + ','
    result_msg += str(args.memory_size) + ','
    result_msg += str(args.key_memory_state_dim) + ','
    result_msg += str(args.value_memory_state_dim) + ','
    result_msg += str(args.summary_vector_output_dim) + ','
    result_msg += str(np.average(aucs)) + ','
    result_msg += str(np.std(aucs)) + ','
    result_msg += str(np.average(accs)) + ','
    result_msg += str(np.std(accs)) + ','
    result_msg += str(np.average(losses)) + ','
    result_msg += str(np.std(losses)) + ','
    result_msg += str(np.average(f1_scores1)) + ','
    result_msg += str(np.average(f1_scores2)) + '\n'
    with open('results/all_result_valid.csv', 'a') as f:
        f.write(result_msg)


if __name__ == '__main__':
    cross_validation()
