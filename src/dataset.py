import torch
import regex as re
from config import *
from augmentation import *
from multiprocessing import Pool, cpu_count


def seq_transformation(raw_data):
    data_output = ''
    if len(raw_data) > 2:
        for word in raw_data.split(" "):
            label = ''
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


def transform_data(file_path, output_path=''):
    if isinstance(file_path, str):
        file_path = [file_path]

    all_data = []
    for file in file_path:
        data_raw = ''
        with open(file, 'r', encoding='utf-8') as f:
            data_raw = f.read()
        data_raw = re.sub(r" \.", ".", data_raw)
        data_raw = re.sub(r" ,", ",", data_raw)
        data_raw = data_raw.split("\n")
        pool = Pool(processes=cpu_count())
        data_raw = pool.map(seq_transformation, data_raw)
        pool.close()
        pool.join()

        all_data.extend(data_raw)

    if not output_path:
        output_path = "/".join(file_path[0].split("/")[:-1]) + "/data_processed"

    with open(output_path, 'w', encoding="utf-8") as f:
        f.write("".join(all_data))

    return all_data


def parse_data(file_path, tokenizer, sequence_len, token_style):
    """

    :param file_path: text file path that contains tokens and punctuations separated by tab in lines
    :param tokenizer: tokenizer that will be used to further tokenize word for BERT like models
    :param sequence_len: maximum length of each sequence
    :param token_style: For getting index of special tokens in config.TOKEN_IDX
    :return: list of [tokens_index, punctuation_index, attention_masks, punctuation_mask], each having sequence_len
    punctuation_mask is used to ignore special indices like padding and intermediate sub-word token during evaluation
    """
    data_items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # loop until end of the entire text
        idx = 0
        while idx < len(lines):
            x = [TOKEN_IDX[token_style]['START_SEQ']]
            y = [0]
            y_mask = [1]  # which positions we need to consider while evaluating i.e., ignore pad or sub tokens

            # loop until we have required sequence length
            # -1 because we will have a special end of sequence token at the end
            while len(x) < sequence_len - 1 and idx < len(lines):

                word, punc = lines[idx].strip().split('\t')
                tokens = tokenizer.tokenize(word)
                # if taking these tokens exceeds sequence length we finish current sequence with padding
                # then start next sequence from this token
                if len(tokens) + len(x) >= sequence_len:
                    break
                else:
                    for i in range(len(tokens) - 1):
                        x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
                        y.append(0)
                        y_mask.append(0)
                    if len(tokens) > 0:
                        x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
                    else:
                        x.append(TOKEN_IDX[token_style]['UNK'])
                    y.append(punctuation_dict[punc])
                    y_mask.append(1)
                    idx += 1
            x.append(TOKEN_IDX[token_style]['END_SEQ'])
            y.append(0)
            y_mask.append(1)
            if len(x) < sequence_len:
                x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(sequence_len - len(x))]
                y = y + [0 for _ in range(sequence_len - len(y))]
                y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]
            attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]
            data_items.append([x, y, attn_mask, y_mask])

    return data_items


class Dataset(torch.utils.data.Dataset):
    def __init__(self, files, tokenizer, sequence_len, token_style, is_train=False, augment_rate=0.1,
                 augment_type='substitute'):
        """

        :param files: single file or list of text files containing tokens and punctuations separated by tab in lines
        :param tokenizer: tokenizer that will be used to further tokenize word for BERT like models
        :param sequence_len: length of each sequence
        :param token_style: For getting index of special tokens in config.TOKEN_IDX
        :param augment_rate: token augmentation rate when preparing data
        :param is_train: if false do not apply augmentation
        """
        if isinstance(files, list):
            self.data = []
            for file in files:
                self.data += parse_data(file, tokenizer, sequence_len, token_style)
        else:
            self.data = parse_data(files, tokenizer, sequence_len, token_style)
        self.sequence_len = sequence_len
        self.augment_rate = augment_rate
        self.token_style = token_style
        self.is_train = is_train
        self.augment_type = augment_type

    def __len__(self):
        return len(self.data)

    def _augment(self, x, y, y_mask):
        x_aug = []
        y_aug = []
        y_mask_aug = []
        for i in range(len(x)):
            r = np.random.rand()
            if r < self.augment_rate:
                AUGMENTATIONS[self.augment_type](x, y, y_mask, x_aug, y_aug, y_mask_aug, i, self.token_style)
            else:
                x_aug.append(x[i])
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

        attn_mask = [1 if token != TOKEN_IDX[self.token_style]['PAD'] else 0 for token in x]
        return x_aug, y_aug, attn_mask, y_mask_aug

    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]
        attn_mask = self.data[index][2]
        y_mask = self.data[index][3]

        if self.is_train and self.augment_rate > 0:
            x, y, attn_mask, y_mask = self._augment(x, y, y_mask)

        x = torch.tensor(x)
        y = torch.tensor(y)
        attn_mask = torch.tensor(attn_mask)
        y_mask = torch.tensor(y_mask)

        return x, y, attn_mask, y_mask
