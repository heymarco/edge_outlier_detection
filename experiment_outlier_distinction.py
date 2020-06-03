import argparse
import os
from src.local_and_global.evaluation import *
from src.data_ import normalize_along_axis, trim_data


if __name__ == '__main__':
    # create data parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str)
    parser.add_argument("-reps", type=int, default=1)
    parser.add_argument("-gpu", type=int)

    # load all files in dir
    args = parser.parse_args()
    dirname = args.data
    reps = args.reps

    # load, trim, normalize data
    data = {}
    ground_truth = {}
    directory = os.path.join(os.getcwd(), "data", dirname)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("d.npy") or file.endswith("o.npy"):
                f = np.load(os.path.join(directory, file))
                f = trim_data(f, max_length=10000)
                if file.endswith("d.npy"):
                    f = normalize_along_axis(f, axis=(0, 1))
                    data[file[:-5]] = f
                if file.endswith("o.npy"):
                    ground_truth[file[:-5]] = f

    # create ensembles
    combinations = [# ("ae", "ae"),
                    # ("ae", "lof"),
                    ("ae", "if"),
                    # ("ae", "xstream")
    ]

    # run ensembles on each data set
    for key in data.keys():
        d = data[key]
        gt = ground_truth[key]
        for c_name, l_name in combinations:
            ensembles = [create_ensembles(d.shape, l_name) for _ in range(reps)]
            results = [train_ensembles(d, ensembles[i], global_epochs=10, l_name=l_name) for i in range(reps)]
            global_scores = [result[0] for result in results]
            local_scores = [result[1] for result in results]
            labels = classify(global_scores, local_scores)
            kappa, f1_global, f1_local = evaluate(labels, gt, contamination=0.01)
            print("***************")
            print("Evaluation:")
            print("kappa_m = {}".format(kappa))
            print("f1_global = {}".format(f1_global))
            print("f1_local = {}".format(f1_local))
            print("***************")
