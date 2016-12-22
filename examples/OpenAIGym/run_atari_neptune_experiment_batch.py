import argparse
import multiprocessing as mp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_yaml', help='The config file of the experiment') # nargs='*' in multi mode
    parser.add_argument('--experiment_neptune_id', help='The id of the ', required=True)

