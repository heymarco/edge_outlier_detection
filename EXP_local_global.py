import argparse
import gc
import logging
import os
import tensorflow as tf

from src.local_and_global.functions import *
from src.training import train_ensembles
from src.utils import setup_machine, normalize_along_axis


def create_datasets(args):
    directory = os.path.join(os.getcwd(), "data", args.data)
    entries = os.listdir(directory)
    for entry in entries:
        if entry.endswith(".npy"):
            os.remove(os.path.join(directory, entry))
    if args.vary == "cont":
        logging.info("Varying contamination with outliers")
        contaminations = [0.01, 0.02, 0.03, 0.05]
        for cont in contaminations:
            cmd_string = "GEN_local_global.py -frac_local {} -frac_global {} -dir {}".format(cont/2.0, cont/2.0, args.data)
            data_generator = os.path.join(os.getcwd(), cmd_string)
            os.system("{} {}".format("python", data_generator))
    elif args.vary == "sf":
        logging.info("Varying subspace fraction")
        sfs = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
        for sf in sfs:
            cmd_string = "GEN_local_global.py -sf {} -dir {}".format(sf, args.data)
            data_generator = os.path.join(os.getcwd(), cmd_string)
            os.system("{} {}".format("python", data_generator))
    elif args.vary == "ratio":
        logging.info("Varying ratio between local and global outliers")
        frac_local = [0.01]
        [frac_local.append(frac_local[-1] + 0.005) for _ in range(6)]
        print(frac_local)
        for fl in frac_local:
            cmd_string = "GEN_local_global.py -frac_local {} -frac_global {} -dir {}".format(fl, 0.05-fl,
                                                                                           args.data)
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
                del f
    logging.info("Finished data loading")
    return data, ground_truth


if __name__ == '__main__':
    # create data parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str,
                        help="The data directory name")  # todo: remove!
    parser.add_argument("-reps", type=int, default=10,
                        help="The number of experiment repetitions")
    parser.add_argument("-gpu", type=int,
                        help="The cuda visible device")
    parser.add_argument("-vary", type=str, choices=["cont", "ratio", "sf"],
                        help="The parameter to vary (each choice corresponds to one of three experiments in the paper)")

    logging.getLogger().setLevel(logging.INFO)
    args = parser.parse_args()
    dirname = args.data
    reps = args.reps

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.compat.v1.enable_eager_execution()

    # select GPU
    setup_machine(cuda_device=args.gpu)

    # create ensembles
    combinations = [("ae", "ae"),
                    ("ae", "lof8"),
                    ("ae", "if"),
                    ("ae", "xstream")
    ]
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
                tf.compat.v1.reset_default_graph()
                ensembles = create_ensembles(d.shape, l_name, contamination=contamination)
                global_scores, local_scores = train_ensembles(d, ensembles, global_epochs=20, l_name=l_name)
                result = [global_scores, local_scores, gt]
                fname = "{}_{}_{}".format(key, c_name, l_name)
                if fname not in results:
                    results[fname] = []
                results[fname].append(result)
                del ensembles
                del result
        del data
        del ground_truth

    target_dir = os.path.join(os.getcwd(), "results", "numpy", args.data)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        os.makedirs(os.path.join(target_dir, "cache"))  # For caching the evaluation as .npy files

    for key in results:
        print(np.array(results[key]).shape)
        np.save(os.path.join(os.getcwd(), "results", "numpy", args.data, key), np.array(results[key]))
