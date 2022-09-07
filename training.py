
import numpy as np
import tensorflow as tf
from tensorflow import keras
from dataset import split_into_train_test

from metrics import compute_performance_metrics_binary
from ploting import PlotLosses
from utils import get_hot_labels


def get_cnn_model(input_shape, n_output_classes, learning_rate):
    """ 
        Returns a 1D CNN model with arch 100 - 50 - GlobalMaxPool1D - 64 - Dropout(0.3) - n_classes. 
        We have used this 1D CNN model extensively in Adversarial research projects.

        Arguments: 
        input_shape (tuple) : Shape of the input
        n_output_classes (int) : number of output classes 
        learning_rate (float) : learning rate for the Adam optimizer

        Returns: 
        A 1D CNN model ready for training, with categorical cross entropy loss and Adam optimizer.
    """
    temp_model = keras.Sequential([
        keras.layers.Conv1D(filters = 100, kernel_size = (10), strides = 2, activation = tf.nn.relu, input_shape = input_shape),
        keras.layers.Conv1D(filters = 50, kernel_size = (5), strides = 1, activation = tf.nn.relu),
        keras.layers.GlobalMaxPool1D(),
        #keras.layers.Flatten(),
        keras.layers.Dense(units = 64, activation = tf.nn.relu),
        keras.layers.Dropout(rate = 0.3),
        keras.layers.Dense(units = n_output_classes, activation = tf.nn.softmax)
    ])
    
    temp_model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(learning_rate=learning_rate), 
                      metrics = ['accuracy'])
    
    return temp_model


def cross_validation_binary_classification(model_function, X, Y, n_CV, test_split, val_split, batch_size=32, epochs=50):
    """
        @brief: Do cross validation for n_CV times and returns the results.

        @param: model_function : A function that returns the model after calling it.
        @param: X (array): Total data
        @param: Y (array): Total label
        @param: test_split (float): The percentage of samples to be included in the test set
        @param: val_split (float): The percentage of samples to be included in the validation set.
        @param: batch_size (int): Default 32
        @param: epochs (int): Default 50

        @return: Results of the cross validation, a dictionary
    """
    x_tr, x_val, x_ts, y_tr, y_val, y_ts = split_into_train_test(X, Y, test_split, val_split=0.0)
    y_tr_hot = get_hot_labels(y_tr)
    y_ts_hot = get_hot_labels(y_ts)

    results_dict = {}
    metrics_arr = []
    for i in range(n_CV):
        model = model_function()
        results = evaluate_model(model, x_tr, y_tr_hot, x_ts, y_ts_hot, validation_split=val_split, 
                                 batch_size=batch_size, epochs=epochs)
        metrics_arr.append(results)
        train_report = compute_performance_metrics_binary(model, x_tr, y_tr)
        test_report = compute_performance_metrics_binary(model, x_ts, y_ts)
        results_dict[i] = {"Training Loss": results[0], "Training Accuracy": results[1], 
                            "Test Loss": results[2], "Test Accuracy": results[3],
                            "Training True Positive": train_report[0], "Training False Positive": train_report[1], 
                            "Training True Negative": train_report[2], "Training False Negative": train_report[3], 
                            "Training Recall": train_report[4], "Training Precision": train_report[5], 
                            "Training F1 Score": train_report[6], "Training ROC AUC": train_report[7],
                            "Training Report": train_report[8],
                            "Test True Positive": test_report[0], "Test False Positive": test_report[1], 
                            "Test True Negative": test_report[2], "Test False Negative": test_report[3], 
                            "Test Recall": test_report[4], "Test Precision": test_report[5], 
                            "Test F1 Score": test_report[6], "Test RO AUC": test_report[7],
                            "Test Report": test_report[8]}

    metrics_arr = np.array(metrics_arr).reshape(n_CV, 4)
    print("Average Training Set Accuracy {:.3f}".format(np.average(metrics_arr[:, 1].ravel())))
    print("Average Testing Set Accuracy {:.3f}".format(np.average(metrics_arr[:, 3].ravel())))

    return results_dict


def train_evaluate_classification_model(model, x_tr, y_tr, x_ts, y_ts, val_split=0.0, 
                   batch_size=32, epochs=50, callbacks=[], 
                   metric_names=['accuracy', 'loss']):
    """
        @brief: Train the model and evaluate it on training and test set and return the results.

        @param: model: TF model
        @param: x_tr: training x
        @param: y_tr: training y
        @param: x_ts: test x
        @param: y_ts: test y
        @param: val_split: validation set split
        @param: BATCH_SIZE (int): default value 32
        @param: EPOCHS (int): default value 50
        @param: callbacks: TF callback functions
        @param: metric_names

        @return: Train and test metrics
    """
    # plot loss function
    plot_loss_cb = PlotLosses(metric_names)
    cbs = [plot_loss_cb]
    
    # append other callbacks
    for c in callbacks:
      cbs.append(c)

    # fit the model
    model_history = model.fit(x_tr, y_tr, batch_size = batch_size, epochs = epochs, 
                              validation_split = val_split, verbose = 0, callbacks = cbs)

    # get the performance values
    train_metrics = model.evaluate(x_tr, y_tr)
    test_metrics = model.evaluate(x_ts, y_ts)
    
    return train_metrics, test_metrics
