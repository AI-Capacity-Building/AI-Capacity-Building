import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
import mlflow.keras



experiment_name = 'MNIST-Classification'
mlflow.set_experiment(experiment_name)

from tensorflow import keras
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_valid = X_train[5000:], X_train[:5000]
y_train, y_valid = y_train[5000:], y_train[:5000]
X_train, X_valid = X_train / 255, X_valid / 255

with mlflow.start_run(run_name='three-layers-network'):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    model.add(keras.layers.Dense(300, activation='relu'))
    model.add(keras.layers.Dense(300, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    loss = 'sparse_categorical_crossentropy'
    optimizer = 'adam'
    metrics = ['accuracy']
    epoch = 10
    batch_size = 128

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    mlflow.keras.autolog()

    model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(X_valid, y_valid))

    test_loss, test_accuracy = model.evaluate(X_test, y_test)