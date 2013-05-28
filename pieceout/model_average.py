import argparse
from theano import config

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=argparse.FileType('r'))
    parser.add_argument('-N', '--num-hiddens', type=int)


if __name__ == "__main__":
    pass
