import argparse
import gc
import glob
import logging
import os

from src.data.synthetic_data import normalize_along_axis
from src.local_and_global.functions import *
from src.training import train_ensembles
from src.utils import setup_machine


def create_datasets(args):
    directory = os.path.join(os.getcwd(), "data", args.data)
    files = glob.glob(os.path.join(directory, "*"))
    for f in files:
        os.remove(f)
    contaminations = [0.002, 0.005, 0.01, 0.05, 0.1]
    for cont in contaminations:
        cmd_string = "GEN_mixed_data.py -frac_local {} -frac_global {} -dir {}".format(cont/2.0, cont/2.0, args.data)
        data_generator = os.path.join(os.getcwd(), cmd_string)
        os.system("{} {}".format("python", data_generator))

    # load, trim, normalize data
    data = {}
    ground_truth = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("d.npy") or file.endswith("o.npy"):
                f = np.load(os.path.join(directory, file))
                if file.endswith("d.npy"):
                    f = normalize_along_axis(f, axis=(0, 1))
                    data[file[:-6]] = f
                if file.endswith("o.npy"):
                    ground_truth[file[:-6]] = f
    logging.info("Finished data loading")
    return data, ground_truth


if __name__ == '__main__':
    # create data parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, default="vary_beta")
    parser.add_argument("-reps", type=int, default=1)
    parser.add_argument("-gpu", type=int)

    logging.getLogger().setLevel(logging.INFO)
    args = parser.parse_args()
    dirname = args.data
    reps = args.reps

    # select GPU
    setup_machine(cuda_device=args.gpu)

    # create ensembles
    combinations = [("ae", "ae"),
                    # ("ae", "lof8"),
                    # ("ae", "if"),
                    # ("ae", "xstream")
    ]
    logging.info("Executing combinations {}".format(combinations))

    results = {}
    for i in range(reps):
        logging.info("Rep {}".format(i))
        data, ground_truth = create_datasets(args)
        for c_name, l_name in combinations:
            for key in data.keys():
                d = data[key]
                gt = ground_truth[key].flatten()
                contamination = np.sum(gt > 0) / len(gt)
                ensembles = create_ensembles(d.shape, l_name, contamination=contamination)
                global_scores, local_scores = train_ensembles(d, ensembles, global_epochs=20, l_name=l_name)
                result = [global_scores, local_scores, gt]
                del ensembles
                gc.collect()
                fname = "{}_{}_{}".format(key, c_name, l_name)
                if i == 0:
                    results[fname] = [result]
                else:
                    results[fname].append(result)

    for key in results:
        np.save(os.path.join(os.getcwd(), "results", "numpy", "vary_beta", key), np.array(results[key]).astype(float))
