
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

def split_into_train_test(X, Y, test_split = 0.25):
    """ 
        Given data (X, Y), split the data into training and testing sets.
        Validation is 10 percent of the training set.

        Arguments:
            X (numpy.ndarray): Data vector
            Y (numpy.ndarray): Label vector
            test_split (float): Test split (0.25 by default)

        Returns:
            x_train, y_train, x_test, and y_test
    """
    if len(X) != len(Y):
        raise ValueError("X and Y must be the same length")
    
    # split the data
    random_state = 42
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_split, random_state=random_state, 
                                                        shuffle=True, stratify=Y)
    
    # x_val = np.array([])
    # y_val = np.array([])
    # if val_split > 0.0:
    #     x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_split, random_state=random_state, 
    #                                                       shuffle=True, stratify=y_train)

    print("Training set {} \nTest set {}".format(x_train.shape, x_test.shape))
    return x_train, x_test, y_train, y_test

def select_random_samples(data, n_samples):
    """
        @brief: Select n_samples random samples from the data
        @param: data (array)
        @param: n_samples (int) Number of samples to randomly select from the data.

        @return: Randomly selected samples
    """
    length = data.shape[0]
    print(length, n_samples)
    if n_samples >= length:
        return data
    else:
        random_index = np.random.randint(low=0, high=length, size=n_samples)
        return data[random_index]


def get_hot_labels(Y):
    """
        Given label vector, return the one hot encoded label vector.

        Arguments:
            Y (numpy.ndarray): label vector
        
        Returns:
            One hot encoded label vector.
    """
    return keras.utils.to_categorical(Y, np.max(Y) + 1, dtype=int)


def create_tf_dataset(X, Y, batch_size, test_size=0.3):
  """ Create train and test TF dataset from X and Y
    The prefetch overlays the preprocessing and model execution of a training step. 
    While the model is executing training step s, the input pipeline is reading the data for step s+1.
    AUTOTUNE automatically tune the number for sample which are prefeteched automatically. 
    
    Keyword arguments:
    X -- numpy array
    Y -- numpy array
    batch_size -- integer
  """
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  
  X = X.astype('float32')
  Y = Y.astype('float32')
  
  x_tr, x_ts, y_tr, y_ts = train_test_split(X, Y, test_size = 0.3, random_state=42, stratify=Y, shuffle=True)
  
  print(f"Train size: {x_tr.shape[0]}")
  print(f"Test size: {x_ts.shape[0]}")

  train_dataset = tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
  train_dataset = train_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
  train_dataset = train_dataset.batch(batch_size).prefetch(AUTOTUNE)
  
  test_dataset = tf.data.Dataset.from_tensor_slices((x_ts, y_ts))
  test_dataset = test_dataset.batch(batch_size).prefetch(AUTOTUNE)
  
  return train_dataset, test_dataset
