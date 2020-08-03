import numpy as np
import tensorflow as tf

from src.utils import average_weights


def train_ensembles(data, ensembles, l_name, global_epochs=10, convolutional=False):
    collab_detectors = ensembles[0]
    local_detectors = ensembles[1]

    oldshape = data.shape
    fshape = (oldshape[0], oldshape[1], oldshape[-3]*oldshape[-2]*oldshape[-1])
    fdata = data if not convolutional else data.reshape(fshape)

    # federated training
    for _ in range(global_epochs):
        collab_detectors = train_federated(models=collab_detectors, data=data, epochs=1, batch_size=32,
                                           frac_available=1.0)
    tf.keras.backend.clear_session()

    # global scores
    predicted = np.array([model.predict(data[i]) for i, model in enumerate(collab_detectors)])
    if convolutional: 
        predicted = predicted.reshape(fshape)
    diff = predicted - fdata
    dist = np.linalg.norm(diff, axis=-1)
    global_scores = dist.flatten()

    print("Fitting {}".format(l_name))
    # local training
    if l_name.startswith("lof") or l_name == "if" or l_name == "xstream":
        if convolutional:
            [l.fit(data[i]) for i, l in enumerate(local_detectors)]
        else:
            [l.fit(fdata[i]) for i, l in enumerate(local_detectors)]
    if l_name == "ae":
        for i, l in enumerate(local_detectors):
            if convolutional:
                l.fit(data[i], data[i], batch_size=32, epochs=global_epochs)
            else:
                l.fit(fdata[i], fdata[i], batch_size=32, epochs=global_epochs)
            tf.keras.backend.clear_session()

    # local scores
    if l_name.startswith("lof"):
        local_scores = - np.array([model.negative_outlier_factor_ for i, model in enumerate(local_detectors)])
    if l_name == "xstream":
        local_scores = np.array([-model.score(fdata[i]) for i, model in enumerate(local_detectors)])
    if l_name == "if":
        local_scores = -np.array([model.score_samples(fdata[i]) for i, model in enumerate(local_detectors)])
    if l_name == "ae":
        if convolutional:
            predicted = np.array([model.predict(data[i]) for i, model in enumerate(local_detectors)])
            predicted = predicted.reshape(fshape)
        else:
            predicted = np.array([model.predict(fdata[i]) for i, model in enumerate(local_detectors)])
        diff = predicted - fdata
        dist = np.linalg.norm(diff, axis=-1)
        local_scores = dist.flatten()  # np.reshape(dist, newshape=(oldshape[0], oldshape[1]))

    return global_scores, local_scores


def train_federated(models, data, epochs=1, batch_size=32, frac_available=1.0, verbose=1):
    num_devices = len(models)
    active_devices = np.random.choice(range(num_devices), int(frac_available * num_devices), replace=False)
    for i in active_devices:
        models[i].fit(data[i], data[i],
                      epochs=epochs,
                      batch_size=batch_size,
                      shuffle=False,
                      verbose=verbose)

    avg = average_weights(models[active_devices])
    [model.set_weights(avg) for model in models]
    return models


def train_separated(models, data, epochs=1, batch_size=32, frac_available=1.0):
    num_devices = len(models)
    active_devices = np.random.choice(range(num_devices), int(frac_available * num_devices), replace=False)
    for i in active_devices:
        for point in data[i]:
            models[i].fit(np.array([point]), np.array([point]),
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=False,
                          verbose=0)
    return models


def train_central(models, data, epochs=1, batch_size=32, frac_available=1.0):
    d = np.reshape(data, newshape=(data.shape[0]*data.shape[1], data.shape[2]))
    models[0].fit(d, d,
                  epochs=epochs,
                  batch_size=batch_size,
                  shuffle=False,
                  verbose=0)
    return models