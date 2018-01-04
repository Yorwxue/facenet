# the program used for checking the readable of images in data set
# run dataset_read_speed.py:
# export PYTHONPATH=<path/to/your/facenet.py>
# python dataset_read_speed.py <path/to/your/dataset>

import argparse
import sys
import time
import numpy as np
from scipy import misc
from datetime import datetime
import os

import facenet


def main(args):
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.expanduser(args.logs_base_dir)
    print('log dir: %s' % log_dir)
    log_file = os.path.join(log_dir, subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)

    dataset = facenet.get_dataset(args.dir)
    paths, _ = facenet.get_image_paths_and_labels(dataset)
    t = np.zeros((len(paths)))
    x = time.time()
    for i, path in enumerate(paths):
        start_time = time.time()
        # with open(path, mode='rb') as f:
        #     _ = f.read()
        try:
            misc.imread(path)
        except:
            print(path)
            with open(log_file, 'a') as fw:
                    fw.write('%s\n' % path)

        duration = time.time() - start_time
        t[i] = duration
        if i % 1000 == 0 or i == len(paths)-1:
            print('File %d/%d  Total time: %.2f  Avg: %.3f  Std: %.3f' % (i, len(paths), time.time()-x, np.mean(t[0:i])*1000, np.std(t[0:i])*1000))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(('--logs_base_dir'), type=str,
                        default='../logs/dataset')
    parser.add_argument('dir', type=str,
        help='Directory with dataset to test')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
