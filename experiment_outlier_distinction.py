import argparse
from src.local_and_global.evaluation import *
from src.local_outliers.evaluation import get_frac_local
from src.data.synthetic_data import normalize_along_axis

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
                if file.endswith("d.npy"):
                    f = normalize_along_axis(f, axis=(0, 1))
                    data[file[:-6]] = f
                if file.endswith("o.npy"):
                    ground_truth[file[:-6]] = f
    print("Finished data loading")

    # create ensembles
    combinations = [("ae", "ae"),
                    ("ae", "lof8"),
                    ("ae", "if"),
                    ("ae", "xstream")
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
                result = np.vstack((global_scores, local_scores, gt))
                results.append(result)
            fname = "{}_{}_{}".format(key, c_name, l_name)
            np.save(os.path.join(os.getcwd(), "results", "numpy", "local_and_global", fname), results)
