import argparse
import gc
import os
import glob

from src.utils import setup_machine
from src.outlying_partitions.functions import *
from src.data.synthetic_data import normalize_along_axis


def create_datasets(args):
    directory = os.path.join(os.getcwd(), "data", dirname)
    files = glob.glob(os.path.join(directory, "*"))
    for f in files: os.remove(f)
    if args.vary == "frac":
        frac_range = [0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        for frac in frac_range:
            data_generator = os.path.join(os.getcwd(), "GEN_outlying_partitions.py -sf {}".format(frac))
            os.system("{} {}".format("python", data_generator))
    if args.vary == "cont":
        frac_range = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for frac in frac_range:
            data_generator = os.path.join(os.getcwd(), "GEN_outlying_partitions.py -cont {}".format(frac))
            os.system("{} {}".format("python", data_generator))
    if args.vary == "shift":
        frac_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
        for frac in frac_range:
            data_generator = os.path.join(os.getcwd(), "GEN_outlying_partitions.py -shift {}".format(frac))
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
    print("Finished data loading")
    return data, ground_truth


if __name__ == '__main__':
    # create data parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-vary", type=str, choices=["frac", "cont", "shift"])
    parser.add_argument("-data", type=str, default="out_part")
    parser.add_argument("-reps", type=int, default=1)
    parser.add_argument("-gpu", type=int)

    # load all files in dir
    args = parser.parse_args()
    dirname = args.data
    reps = args.reps

    # select GPU
    setup_machine(cuda_device=args.gpu)

    data, ground_truth = create_datasets(args)

    # create ensembles
    combinations = [("ae", "ae"),]
    print("Executing combinations {}".format(combinations))

    # run ensembles on each data set
    for key in data.keys():
        d = data[key]
        gt = ground_truth[key]
        labels = np.any(gt, axis=-1)
        np.save(os.path.join(os.getcwd(), "results", "numpy", "out_part", key + "_labels"), labels)
        for c_name, l_name in combinations:
            results = []
            for _ in range(reps):
                models = create_models(d.shape[0], d.shape[-1], compression_factor=0.4)
                result = train_global_detectors(d, models, global_epochs=20)
                results.append(result)
            results = np.array(results)
            fname = "{}_{}_{}".format(key, c_name, l_name)
            np.save(os.path.join(os.getcwd(), "results", "numpy", "out_part", fname), results)
        # remove unneeded data
        data[key] = None
        ground_truth[key] = None
        gc.collect()
