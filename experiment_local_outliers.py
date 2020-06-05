import argparse

from src.data_ import normalize_along_axis, trim_data
from src.local_outliers.evaluation import *
from src.utils import setup_machine

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, choices=["synth", "mhealth", "emotor", "xdk"], nargs="+")
    parser.add_argument("-alg", type=str, choices=["ae", "lof", "xstream", "if"])
    parser.add_argument("-reps", type=int, default=1)
    parser.add_argument("-gpu", type=int)
    args = parser.parse_args()

    # Load, trim and normalize data
    data_names = args.data
    data_dict = {}
    for name in data_names:
        d = load_data(name)
        d = trim_data(d, max_length=10000)
        d = normalize_along_axis(d, axis=(0, 1))
        data_dict[name] = d

    repetitions = args.reps
    alg = args.alg

    for data_name in data_dict.keys():
        data = data_dict[data_name]

        # Fit and predict in local and global setting
        if alg == "ae":
            setup_machine(cuda_device=args.gpu)
            compression_factor = 0.4
            results_global = [fit_predict_autoencoder_global(data, compression_factor) for _ in range(repetitions)]
            results_local = [fit_predict_autoencoder_local(data, compression_factor) for _ in range(repetitions)]

        if alg == "lof":
            results_global = [fit_predict_lof_global(data) for _ in range(repetitions)]
            results_local = [fit_predict_lof_local(data) for _ in range(repetitions)]

        if alg == "xstream":
            k = 50
            nchains = 50
            depth = 10
            results_global = [fit_predict_xstream_global(data, k=k, nchains=nchains, depth=depth) for _ in range(repetitions)]
            results_local = [fit_predict_xstream_local(data, k=k, nchains=nchains, depth=depth) for _ in range(repetitions)]

        if alg == "if":
            results_global = [fit_predict_if_global(data) for _ in range(repetitions)]
            results_local = [fit_predict_if_local(data) for _ in range(repetitions)]

        # Save result to numpy array
        save_result(results_global, name="{}_global_{}_{}".format(data_name, repetitions, alg))
        save_result(results_local, name="{}_local_{}_{}".format(data_name, repetitions, alg))

        frac_local = get_frac_local(result_global=results_global, result_local=results_local)

        f = open(os.path.join(os.getcwd(), "results", "numpy", "local_outliers", "summary.txt"), "a")
        f.write("The fraction of local outliers for {} on {} is {}\n".format(alg, data_name, frac_local))
        f.close()

        print("The fraction of local outliers for {} is {}".format(data_name, frac_local))


