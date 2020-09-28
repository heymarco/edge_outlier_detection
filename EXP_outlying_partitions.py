import argparse
import logging
import tensorflow as tf

from src.utils import setup_machine, normalize_along_axis
from src.outlying_partition.functions import *


def create_datasets(args):
    directory = os.path.join(os.getcwd(), "data", dirname)
    entries = os.listdir(directory)
    for entry in entries:
        if entry.endswith(".npy"):
            os.remove(os.path.join(directory, entry))
    if args.vary == "frac":
        tested_shift = [0.1, 0.15, 0.2, 0.25]
        tested_sf = [0.1, 0.3, 0.6, 1.0]
        combos = []
        for shift in tested_shift:
            for sf in tested_sf:
                combos.append([shift, sf])
        for combo in combos:
            data_generator = os.path.join(os.getcwd(), "GEN_outlying_partitions.py -shift {} -sf {} -dir {}"
                                          .format(combo[0], combo[1], args.data))
            os.system("{} {}".format("python", data_generator))
    if args.vary == "cont":
        frac_range = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for frac in frac_range:
            data_generator = os.path.join(os.getcwd(), "GEN_outlying_partitions.py -cont {} -dir {}".format(frac, args.data))
            os.system("{} {}".format("python", data_generator))
    if args.vary == "shift":
        shift = np.arange(30+1) / 100
        for val in shift:
            data_generator = os.path.join(os.getcwd(), "GEN_outlying_partitions.py -shift {} -dir {}".format(val, args.data))
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
    return data, ground_truth


if __name__ == '__main__':
    # create data parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, default="out_part")  # todo: remove!
    parser.add_argument("-reps", type=int, default=1,
                        help="The number of experiment repetitions")
    parser.add_argument("-gpu", type=int,
                        help="The cuda visible device")
    parser.add_argument("-vary", type=str, choices=["frac", "cont", "shift"],
                        help="The parameter to vary (each choice corresponds to one of three experiments in the paper)")

    logging.getLogger().setLevel(logging.INFO)
    args = parser.parse_args()
    dirname = args.data
    reps = args.reps

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.compat.v1.enable_eager_execution()

    # select GPU
    setup_machine(cuda_device=args.gpu)

    # We only need C, so we do not need to evaluate all local detectors
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
                tf.compat.v1.reset_default_graph()
                models = create_models(d.shape[0], d.shape[-1], compression_factor=0.4)
                result = train_global_detectors(d, models, global_epochs=20)
                fname = "{}_{}_{}".format(key, c_name, l_name)
                if fname not in results:
                    results[fname] = []
                results[fname].append([result.flatten(), gt])
                del models
                del result
        del data
        del ground_truth

    target_dir = os.path.join(os.getcwd(), "results", "numpy", args.data)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        os.makedirs(os.path.join(target_dir, "cache"))  # For caching the evaluation as .npy files

    for key in results:
        np.save(os.path.join(os.getcwd(), "results", "numpy", args.data, key), np.array(results[key]).astype(float))
