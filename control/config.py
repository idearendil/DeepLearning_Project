"""
configuration file
"""

import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dir-result', type=str, default='.')
parser.add_argument('--project-name', type=str, default='proj')
parser.add_argument('--seed-list', type=list, default=[5, 23, 7, 89, 4])
parser.add_argument('--cpu', type=bool, default=False)

parser.add_argument('--model', type=str, default="default_model")

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--lr-init', type=float, default=0.001)

args = parser.parse_args()
args.dir_root = os.getcwd()
