import pickle


def save_pickle(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
