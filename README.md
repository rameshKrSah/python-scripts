# Python Scripts

A collection of Python scripts containing functions that I often use.

## utils.py

Contains the most used utilities functions and classes.

1. `PlotLosses`: Python class to plot accuracy and loss of a model during training.
2. `print_confusion_matric`: Creates a heatmap of a confusion matrix.
3. `get_features_labels_from_df`: Given a pandas dataframe, returns X and Y arrays.
4. `get_cnn_model`: Returns a basic TF 1D CNN model.
5. `save_data`: Given a path and data, save the data as a pickle.
6. `read_data`: Given a path, load the pickle file.
7. `stylize_axis`: Stylizes the axes by removing ticks and spines.
8. `print_metrics`: Print the evalutation metrics.
9. `compute_performane_metrics`: Given a model and (x, y) compute performance metrics.
10. `split_into_train_val_test`: Given (x, y) split into train, test, and validation sets.
11. `select_random_samples`: Select random samples from a given array.
12. `get_hot_labels`: Given an array, returns one-hot encoded array.
13. `find_min_max`: Returns minimum and maximum values.
14. `load_data_with_preprocessing`: Given a path, load the data and min-max scale it.
15. `cross_validation`: Do cross validation from n-times.
16. `evaluate_model`: Train a model and evalute it on training and test datasets.
17. `segment_sensor_reading`: Overlapping window segmentation of an array.
18. `create_tf_dataset`: Given (x, y) create a tensorflow dataset.
