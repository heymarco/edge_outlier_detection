import numpy as np

from src.utils import average_weights


def train_ensembles(data, ensembles, l_name, global_epochs=10):
    collab_detectors = ensembles[0]
    local_detectors = ensembles[1]

    # federated training
    for _ in range(global_epochs):
        collab_detectors = train_federated(models=collab_detectors, data=data, epochs=1, batch_size=32,
                                           frac_available=1.0)

    # global scores
    predicted = np.array([model.predict(data[i]) for i, model in enumerate(collab_detectors)])
    diff = predicted - data
    dist = np.linalg.norm(diff, axis=-1)
    global_scores = dist.flatten()

    print("Fitting {}".format(l_name))
    # local training
    if l_name.startswith("lof") or l_name == "if" or l_name == "xstream":
        [l.fit(data[i]) for i, l in enumerate(local_detectors)]
    if l_name == "ae":
        for i, l in enumerate(local_detectors):
            l.fit(data[i], data[i], batch_size=32, epochs=global_epochs)

    # local scores
    if l_name.startswith("lof"):
        local_scores = - np.array([model.negative_outlier_factor_ for i, model in enumerate(local_detectors)],
                                  dtype=float).flatten()
    if l_name == "xstream":
        local_scores = np.array([-model.score(data[i]) for i, model in enumerate(local_detectors)],
                                dtype=float).flatten()
    if l_name == "if":
        local_scores = -np.array([model.score_samples(data[i]) for i, model in enumerate(local_detectors)],
                                 dtype=float).flatten()
    if l_name == "ae":
        predicted = np.array([model.predict(data[i]) for i, model in enumerate(local_detectors)])
        diff = predicted - data
        dist = np.linalg.norm(diff, axis=-1)
        local_scores = dist.flatten()
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
