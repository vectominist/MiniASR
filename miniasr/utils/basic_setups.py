'''
    File      [ base_setups.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Basic setups. ]
'''

import logging
import random
import numpy as np
import torch


def set_random_seed(seed: int = 7122):
    ''' Set global random seeds. '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def base_args(parser):
    ''' Base arguments to be parsed. '''
    # Running-related
    parser.add_argument('--cpu', action='store_true',
                        help='Using CPU only.')
    parser.add_argument('--seed', type=int,
                        default=7122, help='Set random seed.')
    parser.add_argument('--njobs', type=int, default=2,
                        help='Number of workers.')

    # Logging
    parser.add_argument('--log-file', type=str, default='none',
                        help='Logging file.')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO',
                                 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level.')

    # Debug
    parser.add_argument('--detect-anomaly', action='store_true',
                        help='Detect anomaly with torch.autograd.set_detect_anomaly.')

    return parser


def logging_args(args):
    ''' Set logging config. '''

    level = getattr(logging, args.log_level)

    if args.log_file != 'none':
        logging.basicConfig(
            level=level,
            format='%(asctime)s %(filename)s.%(funcName)s(%(lineno)d) %(message)s',
            datefmt='%Y-%m-%d %H:%M',
            handlers=[logging.FileHandler(args.log_file, 'w', 'utf-8')])
    else:
        logging.basicConfig(
            level=level,
            format='%(asctime)s %(filename)s.%(funcName)s(%(lineno)d) %(message)s',
            datefmt='%Y-%m-%d %H:%M')
