import numpy as np
import third_party.pyeeg.pyeeg as pyeeg


def compute_features(X, feature_names, history, config):
    '''
    This is just wrapper function to PyEEG APIs.
    This function computes PyEEG features for entire dataset.

    :param X: input time-series data. Each row will have one time-series per original features
    :param feature_names: PyEEG features to compute
    :param history: Is used to compute number of features. TODO: pass number of features instead
    :param config: part of the
    :return: computed PyEEG features
    '''
    features = []
    col_count = int(X.shape[1] / history)
    M = config["feature_eng"]["M"]
    r = config["feature_eng"]["r"]
    band = config["feature_eng"]["band"]
    FS = config["feature_eng"]["FS"]
    tau = config["feature_eng"]["tau"]
    DE = config["feature_eng"]["DE"]
    for row in X:
        row_features = []
        for col_i in range(col_count):
            data = row[range(col_i, len(row), col_count)]
            st = np.std(data)
            R = r * st
            for feature in feature_names:
                if feature == 'ap_entropy':
                    f = [pyeeg.ap_entropy(data, M, R)]
                elif feature == 'bin_power':
                    bin_power = pyeeg.bin_power(data, band, FS)
                    f = []
                    f.extend(bin_power[0].tolist())
                    f.extend(bin_power[1].tolist())
                elif feature == 'dfa':
                    f = [pyeeg.dfa(data)]
                elif feature == 'fisher_info':
                    f = [pyeeg.fisher_info(data, tau, DE)]
                elif feature == 'hurst':
                    try:
                        hurst = pyeeg.hurst(data)
                    except:
                        hurst = 0
                    f = [hurst]
                elif feature == 'pfd':
                    f = [pyeeg.pfd(data)]
                elif feature == 'samp_entropy':
                    f = [pyeeg.samp_entropy(data, M, R)]
                elif feature == 'spectral_entropy':
                    bin_power = pyeeg.bin_power(data, band, FS)
                    f = pyeeg.spectral_entropy(data, band, FS, bin_power).tolist()
                elif feature == 'svd_entropy':
                    f = [pyeeg.svd_entropy(data, tau, DE)]
                row_features.extend(f)
        features.append(row_features)
    return np.array(features)
