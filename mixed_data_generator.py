import os
import math
import numpy as np
from src.pipelines import dataset_local_outliers, dataset_global_outliers

# configuration
num_devices = 10
num_data = 1000
dims = 100
subspace_frac = 1.0
frac_outlying_devices = 1.0
frac_outlying_data = 0.01

# create local outliers
gamma = 0.5
delta = 0.2
data_local, labels_local = dataset_local_outliers(num_devices=num_devices,
                                                  n=num_data / 2,
                                                  dims=dims,
                                                  subspace_frac=subspace_frac,
                                                  frac_outlying_devices=frac_outlying_devices,
                                                  frac_outlying_data=frac_outlying_data,
                                                  gamma=gamma,
                                                  delta=delta)

# create global outliers
gamma = 0.4
delta = 0.3
data_global, labels_global = dataset_global_outliers(num_devices=num_devices,
                                                     n=num_data / 2,
                                                     dims=dims,
                                                     subspace_frac=subspace_frac,
                                                     frac_outlying_devices=frac_outlying_devices,
                                                     frac_outlying_data=frac_outlying_data,
                                                     gamma=gamma,
                                                     delta=delta)

# concat data
data = np.concatenate((data_local, data_global), axis=1)

# create labels
labels_local = labels_local.astype(np.int32)
labels_global = labels_global.astype(np.int32)
labels_local[labels_local > 0] = 1
labels_global[labels_global > 0] = 2
labels = np.concatenate((labels_local, labels_global), axis=1)

# shuffle
# for i in range(len(data)):
#     shuffled_indices = np.arange(num_data)
#     np.random.shuffle(shuffled_indices)
#     data[i] = data[i][shuffled_indices]
#     labels[i] = labels[i][shuffled_indices]

# write to file
params_str = "{}_{}_{}_{}_{}_{}_{}_{}_mixed".format(num_devices,
                                                    num_data,
                                                    dims,
                                                    subspace_frac,
                                                    frac_outlying_devices,
                                                    frac_outlying_data,
                                                    gamma,
                                                    delta)
dataname = os.path.join(os.getcwd(), "data", "synth", params_str + "_d")
outname = os.path.join(os.getcwd(), "data", "synth", params_str + "_o")
np.save(dataname, data)
np.save(outname, labels)

print(labels)
print("Num outliers = {}".format((np.sum(labels > 0)/dims)))
