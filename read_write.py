import pickle

def save_to_pickle_data(path, data):
    """
        Given a path and data, save the data to the path as a pickle file.

        Arguments:
        path (string) : file path with .pkl extension
        data : data values; can be a single container or multiple containers
    """
    f = open(path, "wb")
    pickle.dump(data, f)
    f.close()

def read_pickle_data(path, n_vaues=None):
    """
        Given a path, read the file and return the contents.

        Arguments:
        path (string) : File path with .pkl extension
        n_values (int) : Number of containers expected to be read. 
    """
    
    f = open(path, "rb")
    d = pickle.load(f)
    f.close()
    return d