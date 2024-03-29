import time

import torch
import torch.nn as nn
from torch.utils import data
import torch.multiprocessing

import mlflow
import numpy as np
from tqdm import tqdm
from uuid import uuid4

import augmentation
from config import *
from argparser import parse_arguments

from dataset import Dataset
from dataset import cpu_count
from model import DeepPunctuation, DeepPunctuationCRF


torch.multiprocessing.set_sharing_strategy('file_system')   # https://github.com/pytorch/pytorch/issues/11201

args = parse_arguments()

# for reproducibility
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

# tokenizer
if 'berto' in args.pretrained_model:
    tokenizer = MODELS[args.pretrained_model][1].from_pretrained('../models/berto/')
elif 'bertinho' in args.pretrained_model:
    tokenizer = MODELS[args.pretrained_model][1].from_pretrained('../models/bertinho/')
else:
    tokenizer = MODELS[args.pretrained_model][1].from_pretrained(args.pretrained_model)

aug_type = args.augment_type
augmentation.tokenizer = tokenizer
augmentation.sub_style = args.sub_style
augmentation.alpha_sub = args.alpha_sub
augmentation.alpha_del = args.alpha_del

token_style = MODELS[args.pretrained_model][3]
ar = args.augment_rate
sequence_len = args.sequence_length

# Datasets
print("+==================+")
print("| Loading data ... |")
print("+------------------+")
if args.language == 'en':
    train_set = Dataset(os.path.join(args.data_path, 'en/train2012'), data_tokenizer=tokenizer, token_style=token_style,
                        sequence_len=sequence_len, batch_size=args.batch_size, is_train=True,
                        augment_rate=ar, augment_type=aug_type)
    print("\ttrain-set loaded")
    val_set = Dataset(os.path.join(args.data_path, 'en/dev2012'), data_tokenizer=tokenizer, sequence_len=sequence_len,
                      batch_size=args.batch_size, token_style=token_style, is_train=False)
    print("\tvalidation-set loaded")
    test_set_ref = Dataset(os.path.join(args.data_path, 'en/test2011'), data_tokenizer=tokenizer, is_train=False,
                           sequence_len=sequence_len, batch_size=args.batch_size, token_style=token_style)
    test_set_asr = Dataset(os.path.join(args.data_path, 'en/test2011asr'), data_tokenizer=tokenizer, is_train=False,
                           sequence_len=sequence_len, batch_size=args.batch_size, token_style=token_style)
    test_set = [val_set, test_set_ref, test_set_asr]
    print("\ttest-set loaded")
elif args.language == 'gl':
    check_for_data_base('gl')
    data_path = os.path.join(args.data_path, 'gl/train')
    train_set = Dataset(data_path, data_tokenizer=tokenizer, sequence_len=sequence_len, batch_size=args.batch_size,
                        token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type)
    print("\ttrain-set loaded")
    data_path = data_path.replace('gl/train', 'gl/dev')
    val_set = Dataset(data_path, data_tokenizer=tokenizer, sequence_len=sequence_len, batch_size=args.batch_size,
                      token_style=token_style, is_train=False)
    print("\tvalidation-set loaded")
    data_path = data_path.replace('gl/dev', 'gl/test')
    test_set_ref = Dataset(data_path, data_tokenizer=tokenizer, sequence_len=sequence_len, batch_size=args.batch_size,
                           token_style=token_style, is_train=False)
    print("\ttest-set loaded")
    test_set = [test_set_ref]
elif args.language == 'gl_big':
    check_for_data_base('gl_big')
    data_path = os.path.join(args.data_path, 'gl_big/train')
    train_set = Dataset(data_path, data_tokenizer=tokenizer, sequence_len=sequence_len, batch_size=args.batch_size,
                        token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type)
    print("\ttrain-set loaded")
    data_path = data_path.replace('gl_big/train', 'gl_big/dev')
    val_set = Dataset(data_path, data_tokenizer=tokenizer, sequence_len=sequence_len, batch_size=args.batch_size,
                      token_style=token_style, is_train=False)
    print("\tvalidation-set loaded")
    data_path = data_path.replace('gl_big/dev', 'gl_big/test')
    test_set_ref = Dataset(data_path, data_tokenizer=tokenizer, sequence_len=sequence_len, batch_size=args.batch_size,
                           token_style=token_style, is_train=False)
    print("\ttest-set loaded")
    test_set = [test_set_ref]
elif args.language == 'es':
    train_set = Dataset(os.path.join(args.data_path, 'es/train'), data_tokenizer=tokenizer, token_style=token_style,
                        sequence_len=sequence_len, is_train=True, batch_size=args.batch_size,
                        augment_rate=ar, augment_type=aug_type)
    print("\ttrain-set loaded")
    val_set = Dataset(os.path.join(args.data_path, 'es/dev'), data_tokenizer=tokenizer, sequence_len=sequence_len,
                      batch_size=args.batch_size, token_style=token_style, is_train=False)
    print("\tdev-set loaded")
    test_set_ref = Dataset(os.path.join(args.data_path, 'es/test'), data_tokenizer=tokenizer, token_style=token_style,
                           sequence_len=sequence_len, batch_size=args.batch_size, is_train=False)
    test_set = [test_set_ref]
    print("\ttest-set loaded")
else:
    raise ValueError('Incorrect language argument for Dataset')

# Data Loaders
print("+======================+")
print("| Loading the Database |")
print("+----------------------+")
data_loader_params = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': cpu_count()
}
train_loader = torch.utils.data.DataLoader(train_set, **data_loader_params)
val_loader = torch.utils.data.DataLoader(val_set, **data_loader_params)
test_loaders = [torch.utils.data.DataLoader(x, **data_loader_params) for x in test_set]

# logs
uniq_id = str(uuid4()).split("-")[0]
if args.save_path:
    save_path = args.save_path + uniq_id
else:
    date = "_".join(time.asctime().split(" ")[:3])
    save_path = f"exp_{args.language}_{date}_{uniq_id}/"

os.makedirs(save_path, exist_ok=True)
model_save_path = os.path.join(save_path, 'weights.pt')
log_path = os.path.join(save_path, args.name + '_logs_.txt')

# Model
device = torch.device('cpu') if args.cuda == -1 else torch.device('cuda:' + str(args.cuda))

print(F"+=============================+")
print(f"|Loading BERT model using {str(device).upper()}|")
print(F"+=============================+")

if args.use_crf:
    deep_punctuation = DeepPunctuationCRF(args.pretrained_model, freeze_bert=args.freeze_bert, lstm_dim=args.lstm_dim)
else:
    deep_punctuation = DeepPunctuation(args.pretrained_model, freeze_bert=args.freeze_bert, lstm_dim=args.lstm_dim)
deep_punctuation.to(device)

if args.loss_w:
    t_weight = torch.tensor(train_set.tensor_weight, device=device)
else:
    t_weight = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], device=device)

criterion = nn.CrossEntropyLoss(weight=t_weight)
optimizer = torch.optim.Adam(deep_punctuation.parameters(), lr=args.lr, weight_decay=args.decay)


def validate(data_loader):
    """
    :return: validation accuracy, validation loss
    """
    num_iteration = 0
    deep_punctuation.eval()
    # Class Metrics
    tp = np.zeros(1 + len(punctuation_dict), dtype=int)
    fp = np.zeros(1 + len(punctuation_dict), dtype=int)
    fn = np.zeros(1 + len(punctuation_dict), dtype=int)
    cm = np.zeros((len(punctuation_dict), len(punctuation_dict)), dtype=int)
    # Global metrics
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for x, y, att, y_mask in tqdm(data_loader, desc='eval'):
            x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
            y_mask = y_mask.view(-1)
            if args.use_crf:
                y_predict = deep_punctuation(x, att, y)
                loss = deep_punctuation.log_likelihood(x, att, y)
                y_predict = y_predict.view(-1)
                y = y.view(-1)
            else:
                y_predict = deep_punctuation(x, att)
                y = y.view(-1)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                loss = criterion(y_predict, y)
                y_predict = torch.argmax(y_predict, dim=1).view(-1)
            val_loss += loss.item()
            num_iteration += 1
            y_mask = y_mask.view(-1)
            correct += torch.sum(y_mask * (y_predict == y).long()).item()
            total += torch.sum(y_mask).item()
            for i in range(y.shape[0]):
                if y_mask[i] == 0:
                    # since we created this position due to padding or sub-word tokenization, so we can ignore it
                    continue
                cor = y[i]
                prd = y_predict[i]
                if cor == prd:
                    tp[cor] += 1
                else:
                    fn[cor] += 1
                    fp[prd] += 1
                cm[cor][prd] += 1
        # ignore first index which is for no punctuation
        tp[-1] = np.sum(tp[1:])
        fp[-1] = np.sum(fp[1:])
        fn[-1] = np.sum(fn[1:])

        accuracy = correct/total
        precision = tp / (tp + fp) if (tp + fp).any() else 0
        recall = tp / (tp + fn) if (tp + fn).any() else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall).any() else 0

        global_loss = val_loss / num_iteration

    return accuracy, global_loss, np.nan_to_num(precision), np.nan_to_num(recall), np.nan_to_num(f1), cm


def test(data_loader):
    """
    :return: precision[numpy array], recall[numpy array], f1 score [numpy array], accuracy, confusion matrix
    """
    print("Strating Train Phase")
    num_iteration = 0
    deep_punctuation.eval()
    # +1 for overall result
    tp = np.zeros(1+len(punctuation_dict), dtype=int)
    fp = np.zeros(1+len(punctuation_dict), dtype=int)
    fn = np.zeros(1+len(punctuation_dict), dtype=int)
    cm = np.zeros((len(punctuation_dict), len(punctuation_dict)), dtype=int)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, att, y_mask in tqdm(data_loader, desc='test'):
            x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
            y_mask = y_mask.view(-1)
            if args.use_crf:
                y_predict = deep_punctuation(x, att, y)
                y_predict = y_predict.view(-1)
                y = y.view(-1)
            else:
                y_predict = deep_punctuation(x, att)
                y = y.view(-1)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                y_predict = torch.argmax(y_predict, dim=1).view(-1)
            num_iteration += 1
            y_mask = y_mask.view(-1)
            correct += torch.sum(y_mask * (y_predict == y).long()).item()
            total += torch.sum(y_mask).item()
            for i in range(y.shape[0]):
                if y_mask[i] == 0:
                    # we can ignore this because we know there won't be any punctuation in this position
                    # since we created this position due to padding or sub-word tokenization
                    continue
                cor = y[i]
                prd = y_predict[i]
                if cor == prd:
                    tp[cor] += 1
                else:
                    fn[cor] += 1
                    fp[prd] += 1
                cm[cor][prd] += 1
    # ignore first index which is for no punctuation
    tp[-1] = np.sum(tp[1:])
    fp[-1] = np.sum(fp[1:])
    fn[-1] = np.sum(fn[1:])
    precision = tp/(tp+fp) if (tp + fp).any() else 0
    recall = tp/(tp+fn) if (tp + fn).any() else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall).any() else 0

    return np.nan_to_num(precision), np.nan_to_num(recall), np.nan_to_num(f1), correct/total, cm


def train():
    with open(log_path, 'a') as f:
        f.write(str(args)+'\n')

    exp_date = "_".join(time.asctime().split(" ")[:3])
    mlflow.set_tracking_uri('http://0.0.0.0:5000')
    mlflow.set_experiment(f"exp_{args.language}_{exp_date}")

    exp_id = mlflow.tracking.MlflowClient().get_experiment_by_name(f"exp_{args.language}_{date}").experiment_id
    with mlflow.start_run(experiment_id=exp_id, run_name=uniq_id):
        # MLflow Tracking #0
        model_parameters = {"model-name": args.pretrained_model, "seed": args.seed, "language": args.language,
                            "epochs": args.epoch, "learning-rate": args.lr, "sequence-length": args.sequence_length,
                            "batch-size": args.batch_size, "lstm-dim": args.lstm_dim,
                            "loss-weighted": t_weight, "crf": args.use_crf, "weight-decay": args.decay,
                            "gradient-clip": args.gradient_clip,
                            "augment-rate": args.augment_rate, "augment-type": args.augment_type,
                            "alpha-sub": args.alpha_sub, "alpha-del": args.alpha_del,
                            }
        db_characters = {"train-set": len(train_set),
                         "dev-set": len(val_set),
                         "test-set": len(test_set_ref)}

        mlflow.log_params(model_parameters)  # Log a model parameters
        mlflow.log_params(db_characters)  # Log a database characteristics
        # MLflow Tracking - end #

        batch_norm = []
        best_val_acc = -1
        for epoch in range(args.epoch):
            train_loss = 0.0
            train_iteration = 0
            correct = 0
            total = 0

            print("Star Training ...")
            deep_punctuation.train()
            for x, y, att, y_mask in tqdm(train_loader, desc='train'):
                x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
                y_mask = y_mask.view(-1)
                if args.use_crf:
                    loss = deep_punctuation.log_likelihood(x, att, y)
                    # y_predict = deep_punctuation(x, att, y)
                    # y_predict = y_predict.view(-1)
                    # y = y.view(-1)
                else:
                    y_predict = deep_punctuation(x, att)
                    y_predict = y_predict.view(-1, y_predict.shape[2])
                    y = y.view(-1)
                    loss = criterion(y_predict, y)
                    y_predict = torch.argmax(y_predict, dim=1).view(-1)

                    correct += torch.sum(y_mask * (y_predict == y).long()).item()

                optimizer.zero_grad()
                train_loss += loss.item()
                train_iteration += 1
                loss.backward()

                # Doing Gradient clipping, very useful!
                if args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(deep_punctuation.parameters(), max_norm=2.0, norm_type=2)

                # Calculate gradient norms
                for n, layer in enumerate(deep_punctuation.ordered_layers):
                    if n == 2:
                        norm_grad = layer.weight.grad.norm().cpu()
                        batch_norm.append(norm_grad.numpy())

                optimizer.step()

                y_mask = y_mask.view(-1)

                total += torch.sum(y_mask).item()

            train_acc = correct / total
            train_loss /= train_iteration

            log = f'epoch: {epoch}, Train loss: {train_loss}, Train accuracy: {train_acc}'
            # MLflow Tracking#
            train_metrics = {"train_loss": train_loss, "train_accuracy": train_acc, "GradientNorm": np.mean(batch_norm)}
            mlflow.log_metrics(train_metrics, step=epoch + 1)
            # Print in log
            with open(log_path, 'a') as f:
                f.write(log + '\n')
            print(log)

            val_acc, val_loss, val_precision, val_recall, val_f1, val_cm = validate(val_loader)

            log = 'epoch: {}, Val loss: {}, Val accuracy: {}\n'.format(epoch, val_loss, val_acc)
            log_val_metrics = f'Precision: {val_precision}\n' \
                              f'Recall: {val_recall}\n' \
                              f'F1 score: {val_f1}\n'
            # Print log
            with open(log_path, 'a') as f:
                f.write(log)
                f.write(log_val_metrics)
            print(log)
            print(log_val_metrics)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(deep_punctuation.state_dict(), model_save_path)

            # MLflow Tracking #
                val_metrics = {"eval_loss": val_loss, "val_accuracy": val_acc,
                               "P_Lower": val_precision[0], "P_Lower-Comma": val_precision[1],
                               "P_Lower-Period": val_precision[2], "P_All-Capital": val_precision[4],
                               "P_Frits-Capital": val_precision[5], "P_All-Capital-Comma": val_precision[6],
                               "P_All-Capital-Period": val_precision[7], "P_Frits-Capital-Comma": val_precision[9],
                               "P_Frits-Capital-Period": val_precision[10],
                               #
                               "R_Lower": val_recall[0], "R_Lower-Comma": val_recall[1], "R_Lower-Period": val_recall[2],
                               "R_All-Capital": val_recall[4], "R_Frits-Capital": val_recall[5],
                               "R_All-Capital-Comma": val_recall[6], "R_All-Capital-Period": val_recall[7],
                               "R_Frits-Capital-Comma": val_recall[9], "R_Frits-Capital-Period": val_recall[10],
                               #
                               "F1_Lower": val_f1[0], "F1_Lower-Comma": val_f1[1], "F1_Lower-Period": val_f1[2],
                               "F1_All-Capital": val_f1[4], "F1_Frits-Capital": val_f1[5],
                               "F1_All-Capital-Comma": val_f1[6], "F1_All-Capital-Period": val_f1[7],
                               "F1_Frits-Capital-Comma": val_f1[9], "F1_Frits-Capital-Period": val_f1[10],
                               }

            mlflow.log_metrics(val_metrics, step=epoch + 1)

        print('Best validation Acc:', best_val_acc)
        deep_punctuation.load_state_dict(torch.load(model_save_path))
        for loader in test_loaders:
            precision, recall, f1, accuracy, cm = test(loader)
            log = 'Precision: ' + str(precision) + '\n' + 'Recall: ' + str(recall) + '\n' + 'F1 score: ' + str(f1) + \
                  '\n' + 'Accuracy:' + str(accuracy) + '\n' + 'Confusion Matrix' + str(cm) + '\n'
            print(log)
            # MLflow Tracking#
            test_metrics = {"test_acc": accuracy}
            mlflow.log_metrics(test_metrics)
            # Print in log
            with open(log_path, 'a') as f:
                f.write(log)
            log_text = ''
            for i in range(1, 5):
                log_text += str(precision[i] * 100) + ' ' + str(recall[i] * 100) + ' ' + str(f1[i] * 100) + ' '
            with open(log_path, 'a') as f:
                f.write(log_text[:-1] + '\n\n')


if __name__ == '__main__':
    train()
