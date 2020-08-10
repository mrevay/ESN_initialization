import scipy.io as io
import numpy as np
import csv

# from load_WH import create_wienerhammerstein_datasets


def load_data(dataset="Silverbox"):

    if dataset == "Silverbox":

        data = io.loadmat('./datasets/SilverboxFiles/SilverboxFiles/SNLS80mV.mat')

        u_val = data["V1"][0:1, -10000:]
        y_val = data["V2"][0:1, -10000:]

        u_train = data["V1"][0:1, 40500:-10000]
        y_train = data["V2"][0:1, 40500:-10000]

        u_test = data["V1"][0:1, :40500]
        y_test = data["V2"][0:1, :40500]

        train = {"u": u_train, "y": y_train}
        test = {"u": u_test, "y": y_test}
        val = {"u": u_val, "y": y_val}

        return [train, val, test]

    if dataset == "WH":

        # which data set to use
        test_set = 'sweptsine'
        train_set = 'small'

        MCiter = 0

        if test_set == 'multisine':
            test_idx = [2, 4]
        elif test_set == 'sweptsine':
            test_idx = [3, 5]

        # data file direction and name
        if train_set == 'small':
            file_name_train = './datasets/WienerHammersteinFiles/WH_MultisineFadeOut.csv'
        elif train_set == 'big':
            file_name_train = './datasets/WienerHammersteinFiles/WH_SineSweepInput_meas.csv'
        file_name_test = './datasets/WienerHammersteinFiles/WH_TestDataset.csv'

        # initialization
        u = []
        y = []
        u_val = []
        y_val = []
        u_test = []
        y_test = []

        # read the file into variable
        with open(file_name_train, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            line_count = 0
            for row in csv_reader:
                if row == []:
                    continue

                # ignore header line
                if line_count == 0:
                    line_count += 1
                else:
                    # Extract combination of training / validation data
                    if file_name_train == './datasets/WienerHammersteinFiles/WH_SineSweepInput_meas.csv':
                        idx = 100 + MCiter
                        u.append(float(row[idx]))
                        y.append(float(row[2 * idx]))
                        u_val.append(float(row[idx + 1]))
                        y_val.append(float(row[2 * idx + 1]))
                    elif file_name_train == './datasets/WienerHammersteinFiles/WH_MultisineFadeOut.csv':
                        idx = 2
                        if MCiter % 2:
                            idx_add = 0
                        else:
                            idx_add = 1
                        u.append(float(row[idx + idx_add]))
                        y.append(float(row[2 * idx + idx_add]))
                        u_val.append(float(row[idx + 1 - idx_add]))
                        y_val.append(float(row[2 * idx + 1 - idx_add]))

        # read the file into variable
        with open(file_name_test, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            line_count = 0
            for row in csv_reader:
                if row ==[]:
                    continue

                # ignore header line
                if line_count == 0:
                    line_count += 1
                else:
                    # Extract combination of training / validation data
                    u_test.append(float(row[test_idx[0]]))  # use 2,4 for multisine
                    y_test.append(float(row[test_idx[1]]))  # use 3,5 for swept sine

        # convert from list to numpy array
        u_train = np.asarray(u)
        y_train = np.asarray(y)
        u_val = np.asarray(u_val)
        y_val = np.asarray(y_val)
        u_test = np.asarray(u_test)
        y_test = np.asarray(y_test)

        # get correct dimensions
        u_test = u_test[..., None].T
        y_test = y_test[..., None].T
        u_train = u_train[..., None].T
        y_train = y_train[..., None].T
        u_val = u_val[..., None].T
        y_val = y_val[..., None].T

        train = {"u": u_train, "y": y_train}
        test = {"u": u_test, "y": y_test}
        val = {"u": u_val, "y": y_val}

        return [train, val, test]


if __name__ == "__main__":

    dataset_train, dataset_valid, dataset_test = create_wienerhammerstein_datasets()
