import numpy as np

from src.cifar100 import create_cifar100_data
from src.models import create_models, train_federated


num_devices = 100
global_epochs = 2

x, y = create_cifar100_data(num_clients=num_devices)

dims = x.shape[-1]*x.shape[-2]*x.shape[-3]

models = create_models(num_devices=num_devices, dims=dims, compression_factor=0.4)

x_newshape = (x.shape[0], x.shape[1], x.shape[2]*x.shape[3]*x.shape[4])
x = x.reshape(x_newshape)

for epoch in np.arange(global_epochs):
    models = train_federated(models, epochs=1, data=x)

# global scores
predicted = np.array([model.predict(x[i]) for i, model in enumerate(models)])
diff = predicted - x
dist = np.linalg.norm(diff, axis=-1)
global_scores = dist.flatten()

print(global_scores)

with np.unique(y.flatten()) as unique_values:
    accumulated_result = []
    for value in unique_values:
        mean_score = np.mean(global_scores[y.flatten() == value])
        accumulated_result.append(mean_score)
    print(accumulated_result)
