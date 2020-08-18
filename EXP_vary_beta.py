import argparse
import gc
import glob
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
    contaminations = [0.001, 0.005, 0.01, 0.05, 0.1]
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
    print("Finished data loading")
    return data, ground_truth


if __name__ == '__main__':
    # create data parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, default="vary_beta")
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
    combinations = [("ae", "ae"),
                    # ("ae", "lof8"),
                    # ("ae", "if"),
                    # ("ae", "xstream")
    ]
    print("Executing combinations {}".format(combinations))

    # run ensembles on each data set
    for key in data.keys():
        d = data[key]
        gt = ground_truth[key]
        contamination = np.sum(gt > 0)/len(gt.flatten())
        for c_name, l_name in combinations:
            results = []
            for i in range(reps):
                ensembles = create_ensembles(d.shape, l_name, contamination=contamination)
                global_scores, local_scores = train_ensembles(d, ensembles, global_epochs=20, l_name=l_name)
                result = [global_scores, local_scores, gt]
                results.append(result)
                del ensembles
                gc.collect()
            fname = "{}_{}_{}".format(key, c_name, l_name)
            np.save(os.path.join(os.getcwd(), "results", "numpy", "vary_beta", fname), results)
        # remove unneeded data
        data[key] = None
        ground_truth[key] = None
