import torch
import regex as re

from config import *
from augmentation import *
from multiprocessing import Pool, cpu_count


def seq_transformation(raw_data, removelist=',.? '):
    raw_data = re.sub(r'[^\w'+removelist+']', " ", raw_data)
    raw_data = re.sub(r' +', " ", raw_data)
    raw_data = raw_data.strip()
    data_output = ''
    if len(raw_data) > 2:
        for word in raw_data.split(" "):
            label = ''
            if word:
                if word == word.lower():
                    if word.isalnum():
                        label = "\t" + "O"
                    elif word[-1] == ",":
                        label = "\t" + "COMMA"
                    elif word[-1] == ".":
                        label = "\t" + "PERIOD"
                    elif word[-1] == "?":
                        label = "\t" + "QUESTION"
                elif word == word.upper():
                    if word.isalnum():
                        label = "\t" + "ALL_CAPITAL"
                    elif word[-1] == ",":
                        label = "\t" + "ALL_CAPITAL+COMMA"
                    elif word[-1] == ".":
                        label = "\t" + "ALL_CAPITAL+PERIOD"
                    elif word[-1] == "?":
                        label = "\t" + "ALL_CAPITAL+QUESTION"
                elif word[0] == word[0].upper():
                    if word.isalnum():
                        label = "\t" + 'FRITS_CAPITAL'
                    elif word[-1] == ",":
                        label = "\t" + "FRITS_CAPITAL+COMMA"
                    elif word[-1] == ".":
                        label = "\t" + "FRITS_CAPITAL+PERIOD"
                    elif word[-1] == "?":
                        label = "\t" + "FRITS_CAPITAL+QUESTION"
                word = re.sub(r"[.,]", "", word).lower()
                if label != '' and word.isalnum():
                    data_output += word + label + '\n'
    return data_output


def making_datasets(path_data, output_path='', criteria=80, with_eval=True):

    raw_data = []
    if isinstance(path_data, str):
        print("Reading data")
        with open(path_data, 'r') as f:
            raw_data = f.readlines()
    elif not isinstance(path_data, list):
        raise ValueError('Incorrect type for Dataset')
    data_amount = len(raw_data)

    first_cut = int(data_amount * (criteria / 100)) - 1
    train_set = raw_data[:first_cut]
    eval_set, test_set = [], []

    if with_eval:
        print("+---------------------------------------------+")
        print("|   ...Making Train, Eval and Test Sets ...   |")

        second_cut = int(((data_amount - first_cut) / 2))
        eval_set = raw_data[first_cut: (first_cut + second_cut)]
        test_set = raw_data[-second_cut:]

        print("|---------------------------------------------|")
        print(f"| Size of TRAIN-set: {len(train_set)} |")
        print(f"| Size of EVAL-set : {len(eval_set)}  |")
        print(f"| Size of TEST-set : {len(test_set)}  |")
        print("+" + ("-" * (len(f"| Size of TRAIN-set: {len(train_set)} |") - 2)) + "+\n")

        if output_path == '':
            output_path = "/".join(path_data.split("/")[:-1])

        os.makedirs(output_path, exist_ok=True)
        with open(output_path + "/train", 'x') as f:
            f.writelines(train_set)
        with open(output_path + "/dev", 'x') as f:
            f.writelines(eval_set)
        with open(output_path + "/test", 'x') as f:
            f.writelines(test_set)

        return train_set, test_set, eval_set
    else:
        print("+---------------------------------------------+")
        print("|   ...Making Train, and Test Sets ...   |")

        test_set = raw_data[first_cut:]

        print("|---------------------------------------------|")
        print(f"| Size of TRAIN-set: {len(train_set)} |")
        print(f"| Size of TEST-set : {len(test_set)}  |")
        print("+" + ("-" * (len(f"| Size of TRAIN-set: {len(train_set)} |") - 2)) + "+\n")

        with open(output_path + "/train", 'x') as f:
            f.writelines(train_set)
        with open(output_path + "/test", 'x') as f:
            f.writelines(test_set)

        return train_set, test_set, eval_set


def transform_data(file_path, output_path='', file=False):
    if file:
        file_path = [file_path]
    else:
        file_path = [file_path + '/' + f for f in os.listdir(file_path)]

    for file in file_path:
        data_raw = ''  # initialization is needed for multiprocessing
        with open(file, 'r', encoding='utf-8') as f:
            data_raw = f.read()
        data_raw = re.sub(r" \.", ".", data_raw)
        data_raw = re.sub(r" ,", ",", data_raw)
        data_raw = re.sub(r" +", " ", data_raw)
        data_raw = re.sub(r"\n ", "\n", data_raw)
        data_raw = data_raw.split("\n")
        pool = Pool(processes=cpu_count())
        data_raw = pool.map(seq_transformation, data_raw)
        pool.close()
        pool.join()

        if not output_path:
            output_path = "/".join(file_path[0].split("/")[:-1]) + "/data_processed"

        with open(output_path, 'a', encoding="utf-8") as f:
            f.write("".join(data_raw))


def parse_data(file_path, d_tokenizer, sequence_len, token_style):
    """

    :param file_path: text file path that contains tokens and punctuations separated by tab in lines
    :param d_tokenizer: tokenizer that will be used to further tokenize word for BERT like models
    :param sequence_len: maximum length of each sequence
    :param token_style: For getting index of special tokens in config.TOKEN_IDX
    :return: list of [tokens_index, punctuation_index, attention_masks, punctuation_mask], each having sequence_len
    punctuation_mask is used to ignore special indices like padding and intermediate sub-word token during evaluation
    """
    data_items = []
    dict_weight = {}.fromkeys(punctuation_dict.keys(), 0)

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # loop until end of the entire text
        idx = 0
        while idx < len(lines):
            x_ = [TOKEN_IDX[token_style]['START_SEQ']]
            y = [0]
            y_mask = [1]  # which positions we need to consider while evaluating i.e., ignore pad or sub tokens

            # loop until we have required sequence length
            # -1 because we will have a special end of sequence token at the end
            while len(x_) < sequence_len - 1 and idx < len(lines):

                word, punc = lines[idx].strip().split('\t')
                tokens = d_tokenizer.tokenize(word)
                if punc in dict_weight:
                    dict_weight[punc] += 1

                # if taking these tokens exceeds sequence length we finish current sequence with padding
                # then start next sequence from this token
                if len(tokens) + len(x_) >= sequence_len:
                    break
                else:
                    for i in range(len(tokens) - 1):
                        x_.append(d_tokenizer.convert_tokens_to_ids(tokens[i]))
                        y.append(0)
                        y_mask.append(0)
                    if len(tokens) > 0:
                        x_.append(d_tokenizer.convert_tokens_to_ids(tokens[-1]))
                    else:
                        x_.append(TOKEN_IDX[token_style]['UNK'])
                    y.append(punctuation_dict[punc])
                    y_mask.append(1)
                    idx += 1
            x_.append(TOKEN_IDX[token_style]['END_SEQ'])
            y.append(0)
            y_mask.append(1)
            if len(x_) < sequence_len:
                x_ = x_ + [TOKEN_IDX[token_style]['PAD'] for _ in range(sequence_len - len(x_))]
                y = y + [0 for _ in range(sequence_len - len(y))]
                y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]
            attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x_]
            data_items.append([x_, y, attn_mask, y_mask])

    freq = list(sorted(set(dict_weight.values())))
    freq = freq[0] if freq[0] != 0 else freq[1]
    weights = [freq/i if i != 0 else 0 for i in dict_weight.values()]

    return data_items, weights


def aux_calculate_distribution(batch_data_p):
    dict_weight_p = {}.fromkeys(punctuation_dict.keys(), 0)
    for d_p in batch_data_p:
        element_p = d_p.strip().split("\t")
        dict_weight_p[element_p[1]] += 1

    freq_p = list(sorted(set(dict_weight_p.values())))
    freq_p = freq_p[0] if freq_p[0] != 0 else freq_p[1]
    weights_p = np.asanyarray([freq_p / i if i != 0 else 0 for i in dict_weight_p.values()])
    return weights_p


def calculate_distribution(all_batch_data):
    pool = Pool(processes=cpu_count())
    weights_p = pool.map(aux_calculate_distribution, all_batch_data)
    pool.close()
    pool.join()

    return np.asanyarray(weights_p, np.float32).mean(0)


class DatasetAllMemo(torch.utils.data.Dataset):
    def __init__(self, files, data_tokenizer, sequence_len, token_style, is_train=False, augment_rate=0.1,
                 augment_type='substitute'):
        """

        :param files: single file containing tokens and punctuations separated by tab in lines
        :param data_tokenizer: tokenizer that will be used to further tokenize word for BERT like models
        :param sequence_len: length of each sequence
        :param token_style: For getting index of special tokens in config.TOKEN_IDX
        :param augment_rate: token augmentation rate when preparing data
        :param is_train: if false do not apply augmentation
        """
        self.data, self.tensor_weight = parse_data(files, data_tokenizer, sequence_len, token_style)
        self.sequence_len = sequence_len
        self.augment_rate = augment_rate
        self.token_style = token_style
        self.is_train = is_train
        self.augment_type = augment_type

    def __len__(self):
        return len(self.data)

    def _augment(self, x_, y, y_mask):
        x_aug = []
        y_aug = []
        y_mask_aug = []
        for i in range(len(x_)):
            r = np.random.rand()
            if r < self.augment_rate:
                AUGMENTATIONS[self.augment_type](x_, y, y_mask, x_aug, y_aug, y_mask_aug, i, self.token_style)
            else:
                x_aug.append(x_[i])
                y_aug.append(y[i])
                y_mask_aug.append(y_mask[i])

        if len(x_aug) > self.sequence_len:
            # len increased due to insert
            x_aug = x_aug[0:self.sequence_len]
            y_aug = y_aug[0:self.sequence_len]
            y_mask_aug = y_mask_aug[0:self.sequence_len]
        elif len(x_aug) < self.sequence_len:
            # len decreased due to delete
            x_aug = x_aug + [TOKEN_IDX[self.token_style]['PAD'] for _ in range(self.sequence_len - len(x_aug))]
            y_aug = y_aug + [0 for _ in range(self.sequence_len - len(y_aug))]
            y_mask_aug = y_mask_aug + [0 for _ in range(self.sequence_len - len(y_mask_aug))]

        attn_mask = [1 if token != TOKEN_IDX[self.token_style]['PAD'] else 0 for token in x_]
        return x_aug, y_aug, attn_mask, y_mask_aug

    def __getitem__(self, index):
        x_ = self.data[index][0]
        y = self.data[index][1]
        attn_mask = self.data[index][2]
        y_mask = self.data[index][3]

        if self.is_train and self.augment_rate > 0:
            x_, y, attn_mask, y_mask = self._augment(x_, y, y_mask)

        x_ = torch.tensor(x_)
        y = torch.tensor(y)
        attn_mask = torch.tensor(attn_mask)
        y_mask = torch.tensor(y_mask)

        return x_, y, attn_mask, y_mask


class Dataset(torch.utils.data.Dataset):
    def __init__(self, files, data_tokenizer, sequence_len, token_style, batch_size, is_train=False,
                 augment_rate=0.1, augment_type='substitute'):
        """

        :param files: single file containing tokens and punctuations separated by tab in lines
        :param data_tokenizer: tokenizer that will be used to further tokenize word for BERT like models
        :param sequence_len: length of each sequence
        :param token_style: For getting index of special tokens in config.TOKEN_IDX
        :param augment_rate: token augmentation rate when preparing data
        :param is_train: if false do not apply augmentation
        """
        with open(files, 'r', encoding='utf-8') as f:
            self.raw_data = f.readlines()
        chunk_size = sequence_len * batch_size
        chunks = [self.raw_data[i:i + chunk_size] for i in range(0, len(self.raw_data), chunk_size)]
        self.tensor_weight = calculate_distribution(chunks)
        self.tokenizer = data_tokenizer
        self.sequence_len = sequence_len
        self.augment_rate = augment_rate
        self.token_style = token_style
        self.is_train = is_train
        self.augment_type = augment_type

    def __len__(self):
        return len(self.raw_data)//self.sequence_len

    def _augment(self, x_, y, y_mask):
        x_aug = []
        y_aug = []
        y_mask_aug = []
        for i in range(len(x_)):
            r = np.random.rand()
            if r < self.augment_rate:
                AUGMENTATIONS[self.augment_type](x_, y, y_mask, x_aug, y_aug, y_mask_aug, i, self.token_style)
            else:
                x_aug.append(x_[i])
                y_aug.append(y[i])
                y_mask_aug.append(y_mask[i])

        if len(x_aug) > self.sequence_len:
            # len increased due to insert
            x_aug = x_aug[0:self.sequence_len]
            y_aug = y_aug[0:self.sequence_len]
            y_mask_aug = y_mask_aug[0:self.sequence_len]
        elif len(x_aug) < self.sequence_len:
            # len decreased due to delete
            x_aug = x_aug + [TOKEN_IDX[self.token_style]['PAD'] for _ in range(self.sequence_len - len(x_aug))]
            y_aug = y_aug + [0 for _ in range(self.sequence_len - len(y_aug))]
            y_mask_aug = y_mask_aug + [0 for _ in range(self.sequence_len - len(y_mask_aug))]

        attn_mask = [1 if token != TOKEN_IDX[self.token_style]['PAD'] else 0 for token in x_]
        return x_aug, y_aug, attn_mask, y_mask_aug

    def new_parse_data(self, input_data):
        x_, y, attn_mask, y_mask = [], [], [], []
        idx = 0
        # loop until end of the entire text
        while idx < len(input_data):
            x_ = [TOKEN_IDX[self.token_style]['START_SEQ']]
            y = [0]
            y_mask = [1]  # which positions we need to consider while evaluating i.e., ignore pad or sub tokens

            # loop until we have required sequence length
            # -1 because we will have a special end of sequence token at the end
            while len(x_) < self.sequence_len - 1 and idx < len(input_data):
                word, punc = input_data[idx].strip().split('\t')
                word = re.sub(r'[^\w\t\n]', "", word)
                punc = re.sub(r'[^\w\t\n+]', "", punc)
                tokens = self.tokenizer.tokenize(word)
                # if taking these tokens exceeds sequence length we finish current sequence with padding
                # then start next sequence from this token
                if len(tokens) + len(x_) >= self.sequence_len:
                    break
                else:
                    for i in range(len(tokens) - 1):
                        x_.append(self.tokenizer.convert_tokens_to_ids(tokens[i]))
                        y.append(0)
                        y_mask.append(0)

                    if len(tokens) > 0:
                        x_.append(self.tokenizer.convert_tokens_to_ids(tokens[-1]))
                    else:
                        x_.append(TOKEN_IDX[self.token_style]['UNK'])
                    y.append(punctuation_dict[punc])
                    y_mask.append(1)
                    idx += 1
            x_.append(TOKEN_IDX[self.token_style]['END_SEQ'])
            y.append(0)
            y_mask.append(1)
            if len(x_) < self.sequence_len:
                x_ = x_ + [TOKEN_IDX[self.token_style]['PAD'] for _ in range(self.sequence_len - len(x_))]
                y = y + [0 for _ in range(self.sequence_len - len(y))]
                y_mask = y_mask + [0 for _ in range(self.sequence_len - len(y_mask))]
            attn_mask = [1 if token != TOKEN_IDX[self.token_style]['PAD'] else 0 for token in x_]
        return [x_, y, attn_mask, y_mask]

    def __getitem__(self, index):
        if index < len(self.raw_data):
            data_item = self.raw_data[index: index + self.sequence_len]
        else:
            data_item = self.raw_data[index:]

        data_item = self.new_parse_data(data_item)
        x_ = data_item[0]
        y = data_item[1]
        attn_mask = data_item[2]
        y_mask = data_item[3]

        if self.is_train and self.augment_rate > 0:
            x_, y, attn_mask, y_mask = self._augment(x_, y, y_mask)

        x_ = torch.tensor(x_)
        y = torch.tensor(y)
        attn_mask = torch.tensor(attn_mask)
        y_mask = torch.tensor(y_mask)

        return x_, y, attn_mask, y_mask
