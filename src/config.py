import os
import requests
from transformers import *


def check_for_data_base(language=''):

    def download_file_from_google_drive(id_token, dest):
        """
        Download a file from Google Drive using its unique token `id` and write down in disk `destination`.
        :param id_token: Token of the Google Drive file.
        :type id_token: str
        :param dest: path were the Google Drive file will be saved.
        :type dest: str
        :return: -
        """

        def get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value

            return None

        def save_response_content(response, path_to_save):
            chunk_size = 32768
            with open(path_to_save, "wb") as f:
                for chunk in response.iter_content(chunk_size):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)

        url = "https://docs.google.com/uc?export=download"
        session = requests.Session()

        resp = session.get(url, params={'id': id_token}, stream=True)
        token = get_confirm_token(resp)

        if token:
            params = {'id': id_token, 'confirm': token}
            resp = session.get(url, params=params, stream=True)

        save_response_content(resp, dest)

    destination = '../data/' + language + 'train'
    if os.path.exists(destination) is False:
        if language == 'gal':
            file_id = '1z7uA6wwRn0IWCbwcm0gTRi_Rsb6OWndx'
        print("Download TRAIN-SET")
        download_file_from_google_drive(file_id, destination)

    destination = destination.replace('train', 'dev')
    if os.path.exists(destination) is False:
        if language == 'gal':
            file_id = '1AhgwKEk03-9H7cDrbleljRvvEMUCLK2u'
        print("Download DEV-SET")
        download_file_from_google_drive(file_id, destination)

    destination = destination.replace('train', 'test')
    if os.path.exists(destination) is False:
        if language == 'gal':
            file_id = '1W92EQGPk1XKhIRq15fVj_v6NEKGrnLId'
        print("Download TEST-SET")
        download_file_from_google_drive(file_id, destination)



# special tokens indices in different models available in transformers
TOKEN_IDX = {
    'bert': {
        'START_SEQ': 101,
        'PAD': 0,
        'END_SEQ': 102,
        'UNK': 100
    },
    'xlm': {
        'START_SEQ': 0,
        'PAD': 2,
        'END_SEQ': 1,
        'UNK': 3
    },
    'roberta': {
        'START_SEQ': 0,
        'PAD': 1,
        'END_SEQ': 2,
        'UNK': 3
    },
    'albert': {
        'START_SEQ': 2,
        'PAD': 0,
        'END_SEQ': 3,
        'UNK': 1
    },
}

# 'O' -> No punctuation
punctuation_dict = {'O': 0, 'COMMA': 1, 'PERIOD': 2, 'QUESTION': 3, 'ALL_CAPITAL': 4, 'FRITS_CAPITAL': 5,
                    'ALL_CAPITAL+COMMA': 6, 'ALL_CAPITAL+PERIOD': 7, 'ALL_CAPITAL+QUESTION': 8,
                    'FRITS_CAPITAL+COMMA': 9, 'FRITS_CAPITAL+PERIOD': 10, 'FRITS_CAPITAL+QUESTION': 11}

transformation_dict = {0: lambda x: x.lower(), 1: (lambda x: x + ','), 2: (lambda x: x + '.'), 3: (lambda x: x + '?'),
                       4: lambda x: x.upper(), 5: (lambda x: x[0].upper() + x[1:]), 6: (lambda x: x.upper() + ','),
                       7: (lambda x: x.upper() + '.'), 8: (lambda x: x.upper() + '?'),
                       9: (lambda x: x[0].upper() + x[1:] + ','), 10: (lambda x: x[0].upper() + x[1:] + '.'),
                       11: (lambda x: x[0].upper() + x[1:] + '?')}


# pretrained model name: (model class, model tokenizer, output dimension, token style)
MODELS = {
    'bert-base-uncased': (BertModel, BertTokenizer, 768, 'bert'),
    'bert-large-uncased': (BertModel, BertTokenizer, 1024, 'bert'),
    'bert-base-multilingual-cased': (BertModel, BertTokenizer, 768, 'bert'),
    'bert-base-multilingual-uncased': (BertModel, BertTokenizer, 768, 'bert'),
    'xlm-mlm-en-2048': (XLMModel, XLMTokenizer, 2048, 'xlm'),
    'xlm-mlm-100-1280': (XLMModel, XLMTokenizer, 1280, 'xlm'),
    'roberta-base': (RobertaModel, RobertaTokenizer, 768, 'roberta'),
    'roberta-large': (RobertaModel, RobertaTokenizer, 1024, 'roberta'),
    'distilbert-base-uncased': (DistilBertModel, DistilBertTokenizer, 768, 'bert'),
    'distilbert-base-multilingual-cased': (DistilBertModel, DistilBertTokenizer, 768, 'bert'),
    'xlm-roberta-base': (XLMRobertaModel, XLMRobertaTokenizer, 768, 'roberta'),
    'xlm-roberta-large': (XLMRobertaModel, XLMRobertaTokenizer, 1024, 'roberta'),
    'albert-base-v1': (AlbertModel, AlbertTokenizer, 768, 'albert'),
    'albert-base-v2': (AlbertModel, AlbertTokenizer, 768, 'albert'),
    'albert-large-v2': (AlbertModel, AlbertTokenizer, 1024, 'albert'),
    'bertinho-gl-base-cased': (BertModel.from_pretrained('../models/bertinho/'),
                               AutoTokenizer.from_pretrained('../models/bertinho/'), 768, 'bert')
}
