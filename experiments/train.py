# Run a baseline model in BasicTS framework.
# pylint: disable=wrong-import-position
import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

import basicts

torch.set_num_threads(4) # aviod high cpu avg usage

def parse_args():
    parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework!')
    parser.add_argument('-c', '--cfg', default='baselines/Autoformer/METR-LA.py', help='training config')
    parser.add_argument('-g', '--gpus', default='3', help='visible gpus')
    return parser.parse_args()

def parse_args2():
    parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework!')
    parser.add_argument('-c', '--cfg', default='baselines/GLAFF/METR-LA.py', help='training config')
    parser.add_argument('-g', '--gpus', default='3', help='visible gpus')
    return parser.parse_args()

def parse_args3():
    parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework!')
    parser.add_argument('-c', '--cfg', default='baselines/Informer/METR-LA.py', help='training config')
    parser.add_argument('-g', '--gpus', default='3', help='visible gpus')
    return parser.parse_args()

def parse_args4():
    parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework!')
    parser.add_argument('-c', '--cfg', default='baselines/Koopa/METR-LA.py', help='training config')
    parser.add_argument('-g', '--gpus', default='3', help='visible gpus')
    return parser.parse_args()

def parse_args5():
    parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework!')
    parser.add_argument('-c', '--cfg', default='baselines/SparseTSF/METR-LA.py', help='training config')
    parser.add_argument('-g', '--gpus', default='3', help='visible gpus')
    return parser.parse_args()

def parse_args6():
    parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework!')
    parser.add_argument('-c', '--cfg', default='baselines/STAEformer/METR-LA.py', help='training config')
    parser.add_argument('-g', '--gpus', default='3', help='visible gpus')
    return parser.parse_args()

def parse_args7():
    parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework!')
    parser.add_argument('-c', '--cfg', default='baselines/STDDPM/METR-LA.py', help='training config')
    parser.add_argument('-g', '--gpus', default='3', help='visible gpus')
    return parser.parse_args()

def parse_args8():
    parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework!')
    parser.add_argument('-c', '--cfg', default='baselines/UMixer/METR-LA.py', help='training config')
    parser.add_argument('-g', '--gpus', default='3', help='visible gpus')
    return parser.parse_args()

def main():
    args = parse_args()
    basicts.launch_training(args.cfg, args.gpus, node_rank=0)

def main2():
    args = parse_args2()
    basicts.launch_training(args.cfg, args.gpus, node_rank=0)

def main3():
    args = parse_args3()
    basicts.launch_training(args.cfg, args.gpus, node_rank=0)

def main4():
    args = parse_args4()
    basicts.launch_training(args.cfg, args.gpus, node_rank=0)

def main5():
    args = parse_args5()
    basicts.launch_training(args.cfg, args.gpus, node_rank=0)

def main6():
    args = parse_args6()
    basicts.launch_training(args.cfg, args.gpus, node_rank=0)

def main7():
    args = parse_args7()
    basicts.launch_training(args.cfg, args.gpus, node_rank=0)

def main8():
    args = parse_args8()
    basicts.launch_training(args.cfg, args.gpus, node_rank=0)

if __name__ == '__main__':
    main()
    main2()
    main3()
    main4()
    main5()
    main6()
    main7()
    main8()




