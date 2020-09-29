import argparse
import logging
import tensorflow as tf

from src.utils import setup_machine, normalize_along_axis
from src.outlying_partition.functions import *
from GEN_outlying_partitions import create_dataset


def create_datasets(args):
    data = {}
    ground_truth = {}

    if args.vary == "frac":
        tested_shift = [0.1, 0.15, 0.2, 0.25]
        tested_sf = [0.1, 0.3, 0.6, 1.0]
        combos = []
        for shift in tested_shift:
            for sf in tested_sf:
                combos.append([shift, sf])
        for combo in combos:
            shift = combo[0]
            sf = combo[1]
            dataset, labels, params = create_dataset(sigma_p=shift, subspace_frac=sf)
            data[params] = dataset
            ground_truth[params] = dataset

    if args.vary == "shift":
        shift = np.arange(30 + 1) / 100
        for val in shift:
            dataset, labels, params = create_dataset(sigma_p=val)
            data[params] = dataset
            ground_truth[params] = dataset

    # normalize data
    for key in data:
        data[key] = normalize_along_axis(data[key], axis=(0, 1))

    return data, ground_truth


if __name__ == '__main__':
    # create data parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-result_dir", type=str,
                        help="The name of the result directory")
    parser.add_argument("-reps", type=int, default=1,
                        help="The number of experiment repetitions")
    parser.add_argument("-gpu", type=int,
                        help="The cuda visible device")
    parser.add_argument("-vary", type=str, choices=["frac", "cont", "shift"],
                        help="The parameter to vary (each choice corresponds to one of three experiments in the paper)")

    logging.getLogger().setLevel(logging.INFO)
    args = parser.parse_args()
    dirname = args.result_dir
    reps = args.reps

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.compat.v1.enable_eager_execution()

    # select GPU
    setup_machine(cuda_device=args.gpu)

    # We only need C, so we do not need to evaluate all local detectors
    combinations = [("ae", "ae"), ]

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

    target_dir = os.path.join(os.getcwd(), "results", "numpy", args.result_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        os.makedirs(os.path.join(target_dir, "cache"))  # For caching the evaluation as .npy files

    for key in results:
        np.save(os.path.join(target_dir, key), np.array(results[key]).astype(float))
