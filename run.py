import cfgs.config as config
from common.trainer import Trainer
# from common.trainer_bert import Trainer
import argparse, yaml
import random
from easydict import EasyDict as edict
from get_save_vectors import initial_vectors


def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='Bilinear Args')

    parser.add_argument('--run', dest='run_mode',
                        choices=['train', 'val', 'test', 'sample', 'attr'],
                        help='{train, val, test,}',
                        type=str, required=True)

    parser.add_argument('--model', dest='model',
                        choices=['bilinear', 'lstm', 'nsc', 'upnn', 'huapa'],
                        help='{bilinear, ...}',
                        default='bilinear', type=str)

    parser.add_argument('--dataset', dest='dataset',
                        choices=['imdb', 'yelp_13', 'yelp_14', 'digital', 'industrial', 'software'],
                        help='{imdb, yelp_13, yelp_14}',
                        default='imdb', type=str)

    parser.add_argument('--gpu', dest='gpu',
                        help="gpu select, eg.'0, 1, 2'",
                        type=str,
                        default="0, 1")

    parser.add_argument('--seed', dest='seed',
                        help='fix random seed',
                        type=int,
                        default=random.randint(0, 99999999))

    parser.add_argument('--version', dest='version',
                        help='version control',
                        type=str,
                        default="default")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    __C = config.__C

    args = parse_args()
    cfg_file = "cfgs/{}_model.yml".format(args.model)
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.load(f)

    args_dict = edict({**yaml_dict, **vars(args)})
    config.add_edit(args_dict, __C)
    config.proc(__C)

    print('Hyper Parameters:')
    config.config_print(__C)

    # __C.check_path()
    if __C.model == "bilinear":
        initial_vectors(__C.dataset)
        execution = Trainer(__C)
        execution.run(__C.run_mode)
    else:
        exit()
