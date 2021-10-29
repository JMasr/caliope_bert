import os
import re
import torch
import argparse
import itertools
import numpy as np


from tqdm import tqdm
from dataset import Dataset
from model import DeepPunctuation, DeepPunctuationCRF
from config import MODELS, punctuation_dict, transformation_dict


parser = argparse.ArgumentParser(description='Punctuation restoration test')
parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
parser.add_argument('--pretrained-model', default='bertinho-gl-base-cased', type=str, help='pretrained language model')
parser.add_argument('--lstm-dim', default=-1, type=int,
                    help='hidden dimension in LSTM layer, if -1 is set equal to hidden dimension in language model')
parser.add_argument('--use-crf', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to use CRF layer or not')
parser.add_argument('--data-path', default='../data/gl/test', type=str, help='path to test datasets')
parser.add_argument('--weight-path', default='out/weights.pt', type=str, help='model weight path')
parser.add_argument('--sequence_length', default=96, type=int,
                    help='sequence length to use when preparing dataset (default 256)')
parser.add_argument('--batch_size', default=8, type=int, help='batch size (default: 8)')
parser.add_argument('--save-path', default='out/', type=str, help='model and log save directory')

args = parser.parse_args()

# tokenizer
if 'bertinho' in args.pretrained_model:
    tokenizer = MODELS[args.pretrained_model][1].from_pretrained('../models/bertinho/')
else:
    tokenizer = MODELS[args.pretrained_model][1].from_pretrained(args.pretrained_model)
token_style = MODELS[args.pretrained_model][3]

test_set = []
if args.data_path[-1] == "/":
    test_files = os.listdir(args.data_path)
    for file in test_files:
        test_set.append(Dataset(os.path.join(args.data_path, file), tokenizer=tokenizer,
                                sequence_len=args.sequence_length, token_style=token_style, is_train=False))
else:
    test_set.append(Dataset(args.data_path, tokenizer=tokenizer, sequence_len=args.sequence_length,
                            token_style=token_style, is_train=False))

# Data Loaders
data_loader_params = {
    'batch_size': args.batch_size,
    'shuffle': False,
    'num_workers': 0
}

test_loaders = [torch.utils.data.DataLoader(x, **data_loader_params) for x in test_set]

# logs
model_save_path = args.weight_path
log_path = os.path.join(args.save_path, 'logs_test.txt')

# Model
device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
if args.use_crf:
    deep_punctuation = DeepPunctuationCRF(args.pretrained_model, freeze_bert=False, lstm_dim=args.lstm_dim)
else:
    deep_punctuation = DeepPunctuation(args.pretrained_model, freeze_bert=False, lstm_dim=args.lstm_dim)
deep_punctuation.to(device)


def test(data_loader, path_to_model=model_save_path):
    """
    :return: precision[numpy array], recall[numpy array], f1 score [numpy array], accuracy, confusion matrix
    """
    x_str = []
    num_iteration = 0

    deep_punctuation.load_state_dict(torch.load(path_to_model))
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

            # useful inference
            x_tokens = tokenizer.convert_ids_to_tokens(x.view(-1))

            num_iteration += 1
            y_mask = y_mask.view(-1)
            correct += torch.sum(y_mask * (y_predict == y).long()).item()
            total += torch.sum(y_mask).item()
            for i in range(y.shape[0]):
                if y_mask[i] == 0:
                    # we can ignore this because we know there won't be any punctuation in this position
                    # since we created this position due to padding or sub-word tokenization
                    x_tokens[i] = transformation_dict[y_predict[i].item()](x_tokens[i])
                    continue
                cor = y[i]
                prd = y_predict[i]
                if cor == prd:
                    tp[cor] += 1
                else:
                    fn[cor] += 1
                    fp[prd] += 1
                cm[cor][prd] += 1
                x_tokens[i] = transformation_dict[y_predict[i].item()](x_tokens[i])
            x_tokens = " ".join(x_tokens)
            x_tokens = re.sub(r"[£¢]", "", x_tokens)
            x_tokens = re.sub(" ##", "", x_tokens)
            x_tokens = re.sub(r"\[PAD]", "", x_tokens)
            x_tokens = re.sub(r"  +", " ", x_tokens)
            x_str.append(x_tokens)
    # ignore first index which is for no punctuation
    tp[-1] = np.sum(tp[1:])
    fp[-1] = np.sum(fp[1:])
    fn[-1] = np.sum(fn[1:])
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, correct/total, cm, x_str


def inference(data_loader, path_to_model=model_save_path):
    """
    :return: precision[numpy array], recall[numpy array], f1 score [numpy array], accuracy, confusion matrix
    """
    y_str = []
    y_conf = []
    num_iteration = 0

    deep_punctuation.load_state_dict(torch.load(path_to_model))
    deep_punctuation.eval()

    with torch.no_grad():
        for x, y, att, y_mask in tqdm(data_loader, desc='test'):
            x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
            if args.use_crf:
                y_predict = deep_punctuation(x, att, y)
                logits = torch.nn.functional.softmax(y_predict, dim=1)
                y_predict = y_predict.view(-1)
                y = y.view(-1)
            else:
                y_predict = deep_punctuation(x, att)
                logits = torch.nn.functional.softmax(y_predict, dim=1)
                y = y.view(-1)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                y_predict = torch.argmax(y_predict, dim=1).view(-1)

            batch_conf = []
            for b in range(logits.size()[0]):
                for s in range(logits.size()[1]):
                    batch_conf.append(torch.max(logits[b, s, :]).item())

            num_iteration += 1
            x_tokens = tokenizer.convert_ids_to_tokens(x.view(-1))
            for i in range(y.shape[0]):
                x_tokens[i] = transformation_dict[y_predict[i].item()](x_tokens[i])
            y_str.append(x_tokens)
            y_conf.append(batch_conf)

    y_str = list(itertools.chain.from_iterable(y_str))
    y_conf = list(itertools.chain.from_iterable(y_conf))

    ind = 0
    new_text, new_confidence = [], []
    while ind < len(y_str) - 1:
        if y_str[ind] in ['£', '¢', '[pad]', '[PAD]']:
            ind += 1
            continue
        elif (ind != 0) and ("#" in y_str[ind]):
            new_text[-1] = new_text[-1] + y_str[ind][2:]
            new_confidence[-1] = max(y_conf[ind], y_conf[ind - 1])
            ind += 1
            continue
        elif (ind != len(y_str) - 1) and ("#" in y_str[ind + 1]):
            new_t = y_str[ind] + y_str[ind + 1][2:]
            new_c = max(y_conf[ind], y_conf[ind + 1])
            ind += 2
        else:
            new_t = y_str[ind]
            new_c = y_conf[ind]
            ind += 1

        new_text.append(new_t)
        new_confidence.append(new_c)

    return new_text, new_confidence


def run_test():
    for i in range(len(test_loaders)):
        precision, recall, f1, accuracy, cm, text = test(test_loaders[i])
        log = 'Precision: ' + str(precision) + '\n' + 'Recall: ' + str(recall) + '\n' + \
            'F1 score: ' + str(f1) + '\n' + 'Accuracy:' + str(accuracy) + '\n' + 'Confusion Matrix' + str(cm) + '\n'
        print(log)
        with open(log_path, 'a') as f:
            f.write(log)


def run_inference():
    text, confidence = [], []
    for i in range(len(test_loaders)):
        text, confidence = inference(test_loaders[i])

    return text, confidence


run_inference()
run_test()
