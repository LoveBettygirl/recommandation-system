import re
import os
import sys
import getopt
import psutil
from algorithm import SVD_improved
from guppy import hpy

def show_usage():
    print(sys.argv[0], '[opts] [args]')
    print('[opts]: Arguments listed as follows')
    print('[args]: Argument value')
    print('-g: divide train and test set again')
    print('-t: SVD train again')
    print('-h / --help: Show help info')
    print('-m / --memory: Print heap memory detail')
    print('-f / --factors: SVD factor count')
    print('-e / --epochs: SVD epoch count')
    print('-r / --learnrate: SVD learn rate')
    print('-l / --lambda: SVD regularization parameter')
    print('-s / --scale: rate scale')
    print('--trainfile: train file')
    print('--testfile: test file')
    print('--attrfile: attr file')
    sys.exit(1)

def process_memory_usage():
    process = psutil.Process(os.getpid())
    mem = (process.memory_info().rss) / float(2 ** 20)
    return mem

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hmgtf:e:r:l:s:", ["help", "memory", "trainfile=", "testfile=", "attrfile=", "factors=", "epochs=", "learnrate=", "lambda=", "scale="])
    except getopt.GetoptError:
        show_usage()

    svd = SVD_improved.SVD_improved()
    svd.train_file = 'data-new/train.txt'
    svd.test_file = 'data-new/test.txt'
    svd.attr_file = 'data-new/itemAttribute.txt'
    memory = False

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            show_usage()
        elif opt in ("-m", "--memory"):
            memory = True
        elif opt in ("-f", "--factors"):
            svd.factors = int(arg)
        elif opt in ("-e", "--epochs"):
            svd.epochs = int(arg)
        elif opt in ("-r", "--learnrate"):
            svd.learn_rate = float(arg)
        elif opt in ("-l", "--lambda"):
            svd.lambda_ = float(arg)
        elif opt in ("-s", "--scale"):
            pattern = re.compile(r'\d+\,\d+')
            if re.match(pattern, arg) == None:
                show_usage()
            low, high = arg.spilt(',')
            svd.scale = (int(low), int(high))
        elif opt == "--trainfile":
            svd.train_file = arg
        elif opt == "--testfile":
            svd.test_file = arg
        elif opt == "--attrfile":
            svd.attr_file = arg
        elif opt == "-g":
            svd.do_divide = True
        elif opt == "-t":
            svd.do_train = True

    h = None
    if memory:
        h = hpy()
    svd.stat()
    svd.test_model()
    svd.predict()
    print('Process memory usage: %f MB' % process_memory_usage())
    print('Total time cost: %fs' % svd.all_time)
    if memory:
        heap = h.heap()
        print(heap)

if __name__ == "__main__":
    main(sys.argv[1:])