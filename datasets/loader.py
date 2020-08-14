import scipy.io as io
import numpy as np
import csv

# from load_WH import create_wienerhammerstein_datasets


def load_data(dataset="Silverbox"):

    if dataset == "Silverbox":

        data = io.loadmat('./datasets/SilverboxFiles/SilverboxFiles/SNLS80mV.mat')

        u_val = data["V1"][0:1, -10000:].T
        y_val = data["V2"][0:1, -10000:].T

        u_train = data["V1"][0:1, 40500:-10000].T
        y_train = data["V2"][0:1, 40500:-10000].T

        u_test = data["V1"][0:1, :40500].T
        y_test = data["V2"][0:1, :40500].T

        train = {"u": u_train[:, :, None], "y": y_train[:, :, None]}
        test = {"u": u_test[:, :, None], "y": y_test[:, :, None]}
        val = {"u": u_val[:, :, None], "y": y_val[:, :, None]}

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
        u_test = u_test[:, None, None]
        y_test = y_test[:, None, None]
        u_train = u_train[:, None, None]
        y_train = y_train[:, None, None]
        u_val = u_val[:, None, None]
        y_val = y_val[:, None, None]

        train = {"u": u_train, "y": y_train}
        test = {"u": u_test, "y": y_test}
        val = {"u": u_val, "y": y_val}

        return [train, val, test]

    elif dataset == "gait_prediction":
        subject = 1
        val_set = 0

        folder = "./datasets/gait_prediction/python_data/".format(subject)
        file_name = "sub{:d}_Upstairs_canes_all.mat".format(subject)

        data = io.loadmat(folder + file_name)

        # What even is this data format ....
        transpose = lambda x: x.T
        inputs = list(map(transpose, data["p_data"][0, 0][0][0]))
        outputs = list(map(transpose, data["p_data"][0, 0][1][0]))

        # split data into training, validation and test
        L = inputs.__len__()

        mu_u = np.mean(inputs[0], axis=1, keepdims=True)
        mu_y = np.mean(outputs[0], axis=1, keepdims=True)

        scale_u = np.max(inputs[0], axis=1, keepdims=True) - np.min(inputs[0], axis=1, keepdims=True)
        scale_y = np.max(outputs[0], axis=1, keepdims=True) - np.min(outputs[0], axis=1, keepdims=True)

        normalize_u = lambda x: (x - mu_u) / scale_u
        normalize_y = lambda x: (x - mu_y) / scale_y

        # test sets
        u_test = [normalize_u(inputs[-1])]
        y_test = [normalize_y(outputs[-1])]

        # val sets
        u_val = [normalize_u(x) for i, x in enumerate(inputs) if i == val_set]
        y_val = [normalize_y(x) for i, x in enumerate(outputs) if i == val_set]

        # training sets
        u_train = [normalize_u(x) for i, x in enumerate(inputs) if i != val_set and i < L - 1]
        y_train = [normalize_y(x) for i, x in enumerate(outputs) if i != val_set and i < L - 1]

        train = {"u": u_train, "y": y_train}
        test = {"u": u_test, "y": y_test}
        val = {"u": u_val, "y": y_val}

        return train, val, test

    elif dataset == "F16":
        f16_dir = './datasets/F16GVT_Files/BenchmarkData'
        ms1 = io.loadmat(f16_dir + '/F16Data_FullMSine_Level1.mat')
        ms2 = io.loadmat(f16_dir + '/F16Data_FullMSine_Level2_Validation.mat')
        ms3 = io.loadmat(f16_dir + '/F16Data_FullMSine_Level3.mat')
        ms4 = io.loadmat(f16_dir + '/F16Data_FullMSine_Level4_Validation.mat')
        ms5 = io.loadmat(f16_dir + '/F16Data_FullMSine_Level5.mat')
        ms6 = io.loadmat(f16_dir + '/F16Data_FullMSine_Level6_Validation.mat')
        ms7 = io.loadmat(f16_dir + '/F16Data_FullMSine_Level7.mat')

        u_test = np.stack((ms2["Force"], ms4["Force"], ms6["Force"]), 2)
        u_test = u_test.transpose([1, 0, 2])

        y_test = np.stack((ms2["Acceleration"], ms4["Acceleration"], ms6["Acceleration"]), 2)
        y_test = y_test.transpose([1, 0, 2])

        u_train = np.stack((ms1["Force"], ms3["Force"], ms5["Force"], ms7["Force"]), 2)
        u_train = u_train.transpose([1, 0, 2])

        y_train = np.stack((ms1["Acceleration"], ms3["Acceleration"], ms5["Acceleration"], ms7["Acceleration"]), 2)
        y_train = y_train.transpose([1, 0, 2])

        u_val = u_train
        y_val = y_train

        train = {"u": u_train, "y": y_train}
        test = {"u": u_test, "y": y_test}
        val = {"u": u_val, "y": y_val}

        return train, val, test

    elif dataset == "F16_random_grid":
        f16_dir = './datasets/F16GVT_Files/BenchmarkData'
        train_dat = io.loadmat(f16_dir + '/F16Data_SpecialOddMSine_Level2.mat')
        test_dat = io.loadmat(f16_dir + '/F16Data_SpecialOddMSine_Level2_Validation.mat')

        # Test data
        u_test = test_dat["Force"][:, :, None]
        u_test = u_test.transpose([1, 0, 2])

        y_test = test_dat["Acceleration"]
        y_test = y_test.transpose([2, 0, 1])

        # Training data
        u_train = train_dat["Force"][:, None, :]
        u_train = u_train.transpose([2, 1, 0])

        y_train = train_dat["Acceleration"]
        y_train = y_train.transpose([2, 0, 1])

        # Validation
        u_val = u_train[:, :, -1:]
        y_val = y_train[:, :, -1:]

        train = {"u": u_train, "y": y_train}
        test = {"u": u_test, "y": y_test}
        val = {"u": u_val, "y": y_val}

        return train, val, test

if __name__ == "__main__":

    train, val, test = load_data(dataset="F16_random_grid")

    print('~fin~')
