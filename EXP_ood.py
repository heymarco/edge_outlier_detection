import argparse
import gc
import logging
import tensorflow as tf

from src.utils import setup_machine
from src.ood.functions import *
from src.data.synthetic_data import normalize_along_axis


def create_datasets(args):
    directory = os.path.join(os.getcwd(), "data", dirname)
    entries = os.listdir(directory)
    for entry in entries:
        if entry.endswith(".csv"):
            os.remove(os.path.join(directory, entry))
    if args.vary == "frac":
        frac_range = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.13, 0.21, 0.34, 0.55, 0.89, 1.0]
        for frac in frac_range:
            data_generator = os.path.join(os.getcwd(), "GEN_outlying_partitions.py -sf {} -dir {}".format(frac, args.data))
            os.system("{} {}".format("python", data_generator))
    if args.vary == "cont":
        frac_range = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for frac in frac_range:
            data_generator = os.path.join(os.getcwd(), "GEN_outlying_partitions.py -cont {} -dir {}".format(frac, args.data))
            os.system("{} {}".format("python", data_generator))
    if args.vary == "shift":
        frac_range = np.arange(15+1) / 100
        for frac in frac_range:
            data_generator = os.path.join(os.getcwd(), "GEN_outlying_partitions.py -shift {} -dir {}".format(frac, args.data))
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
                del f
    print("Finished data loading")
    print(data)
    return data, ground_truth


if __name__ == '__main__':
    # create data parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-vary", type=str, choices=["frac", "cont", "shift"])
    parser.add_argument("-data", type=str, default="out_part")
    parser.add_argument("-reps", type=int, default=1)
    parser.add_argument("-gpu", type=int)

    logging.getLogger().setLevel(logging.INFO)
    args = parser.parse_args()
    dirname = args.data
    reps = args.reps

    # select GPU
    setup_machine(cuda_device=args.gpu)

    # create ensembles
    combinations = [("ae", "ae"),]

    logging.info("Executing combinations {}".format(combinations))
    logging.info("Repeating {} times".format(reps))

    results = {}
    for i in range(reps):
        logging.info("Rep {}".format(i))
        data, ground_truth = create_datasets(args)
        for c_name, l_name in combinations:
            for key in data.keys():
                d = data[key]
                gt = ground_truth[key].flatten()
                contamination = np.sum(gt > 0) / len(gt)
                tf.keras.backend.clear_session()
                models = create_models(d.shape[0], d.shape[-1], compression_factor=0.4)
                result = train_global_detectors(d, models, global_epochs=20)
                fname = "{}_{}_{}".format(key, c_name, l_name)
                if fname not in results:
                    results[fname] = []
                results[fname].append([result.flatten(), gt])
                del models
                del result
                for _ in range(10):
                    gc.collect()
        del data
        del ground_truth
        for _ in range(10):
            gc.collect()

    for key in results:
        np.save(os.path.join(os.getcwd(), "results", "numpy", args.data, key), np.array(results[key]).astype(float))
