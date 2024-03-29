import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Punctuation restoration')
    # General arguments about the experiment (id, language, gpu-cpu, seed, useful directories
    parser.add_argument('--name', default='caliope_00', type=str, help='name of run')
    parser.add_argument('--language', default='gl', type=str, help='language, available options are gl, es, en')
    parser.add_argument('--cuda', default=0, type=int, help='-1 use cpu positive ints use cuda device')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--data-path', default='../data/', type=str, help='path to train/dev/test datasets')
    parser.add_argument('--save-path', default='', type=str, help='model and log save directory')

    parser.add_argument('--pretrained-model', default='bertinho', type=str, help="pretrained BERT's LM")
    parser.add_argument('--freeze-bert', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Freeze BERT layers or not')
    parser.add_argument('--use-crf', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to use CRF layer or not')

    parser.add_argument('--epoch', default=25, type=int, help='total epochs (default: 10)')
    parser.add_argument('--sequence-length', default=256, type=int,
                        help='sequence length to use when preparing dataset (default 256)')
    parser.add_argument('--batch-size', default=8, type=int, help='batch size (default: 8)')
    parser.add_argument('--lstm-dim', default=-1, type=int,
                        help='hidden dimension in LSTM layer, if -1 is set equal to hidden dimension in language model')

    parser.add_argument('--lr', default=5e-7, type=float, help='learning rate')
    parser.add_argument('--decay', default=0, type=float, help='weight decay (default: 0)')
    parser.add_argument('--gradient-clip', default=1, type=float, help='gradient clipping (default: -1 i.e., none)')
    parser.add_argument('--loss-w', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to use a weighted tensor for loss function')

    parser.add_argument('--augment-rate', default=0., type=float, help='token augmentation probability')
    parser.add_argument('--augment-type', default='none', type=str, help='which augmentation to use')
    parser.add_argument('--sub-style', default='unk', type=str, help='replacement strategy for substitution augment')
    parser.add_argument('--alpha-sub', default=0.4, type=float, help='augmentation rate for substitution')
    parser.add_argument('--alpha-del', default=0.4, type=float, help='augmentation rate for deletion')

    args = parser.parse_args()
    return args
