import json
import numpy as np

import scipy, sklearn, scipy.sparse.csgraph
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler

# load dataset
def load_data(filename, data_separation=None):
    print("Loading the {} data".format(filename))
    file = open(filename, 'r', encoding='utf-8')
    data = json.load(file)
    file.close()
    if data_separation is not None:
        if len(data_separation) == 3:
            return separating_dataset(data, data_separation)
    else:
        return constructing_dataset(data, False, 1)[:7]

# split dataset into train dataset, validate dataset, test dataset
def separating_dataset(dataset, data_separation):
    # Separating dataset to train, validate and test.
    train_count, val_count, test_count = round(len(dataset) * data_separation[0]), \
                                         round(len(dataset) * data_separation[1]), \
                                         len(dataset) - round(len(dataset) * data_separation[0]) - round(
                                             len(dataset) * data_separation[1])
    labels_dic, classes = {}, []
    for k in dataset:
        label = dataset[k][0]
        if labels_dic.get(label) == None:
            labels_dic[label] = 1
        else:
            labels_dic[label] += 1
    # assert len(labelsDic)==2

    for k in labels_dic:
        one_class = [k,
                     round(labels_dic[k] * data_separation[0]),
                     round(labels_dic[k] * data_separation[1]),
                     int(labels_dic[k]) - round(labels_dic[k] * data_separation[0]) - round(
                         labels_dic[k] * data_separation[1])]
        classes.append(one_class)
    classes = np.array(classes)

    print("Train datasets:")
    print(" classes train_num val_num test_num total_num")

    for i in range(0, len(classes)):
        print("{0: ^8}{1: ^10}{2: ^9}{3: ^9}{4: ^9}".format(classes[i][0], classes[i][1], classes[i][2], classes[i][3],
                                                            sum(classes[i, 1:])))
    print("{0: ^8}{1: ^10}{2: ^9}{3: ^9}{4: ^9}".format(" total", sum(classes[:, 1]), sum(classes[:, 2]),
                                                        sum(classes[:, 3]), classes[:, 1:].sum()))

    train_dic, val_dic, test_dic = {}, {}, {}
    for k in dataset:
        label = dataset[k][0]
        index = np.argwhere(classes[:, 0] == label)[0][0].astype(np.int64)
        if (classes[index][1] > 0):
            train_dic[k] = dataset[k]
            classes[index][1] = classes[index][1] - 1
        elif (classes[index][2] > 0):
            val_dic[k] = dataset[k]
            classes[index][2] = classes[index][2] - 1
        else:
            test_dic[k] = dataset[k]
            classes[index][3] = classes[index][3] - 1

    train_data = constructing_dataset(train_dic, True, 1)

    return train_data[:5], \
           constructing_dataset(val_dic, False, 2, train_data[2], train_data[3])[:4], \
           constructing_dataset(test_dic, False, 3, train_data[2], train_data[3])[:4]

# constructing input data for svm
def constructing_dataset(dataset, is_oversampling, data_type, mean_feature=0, std_feature=1):
    # dataset {key,[label,pointlist]}
    if len(dataset) < 1:
        return None
    dataset_resample = {}
    if is_oversampling is True:
        # oversampling
        print("train dataset oversampling...")
        x, y = [], []
        for k in dataset:
            # get the label, vertices coords and features of this sample.
            [label, vertice_coords, vertice_features] = dataset[k]
            x.append(k)
            y.append(label)

        sampler = RandomOverSampler(random_state=0)
        x_resampled, y_resampled = sampler.fit_resample(np.array(x).reshape(-1, 1), y)
        dataset_resample = {}
        for i in range(0, len(x_resampled)):
            k = x_resampled[i][0]
            label = y_resampled[i]
            dataset_resample[str(i)] = dataset[k]

        # print(len(dataset_resample))
        print("Train datasets oversampling:")
        print(" classes train_num total_num")
        labels_dic_oversampling, classes_oversampling = {}, []
        for k in dataset_resample:
            label = dataset_resample[k][0]
            if labels_dic_oversampling.get(label) == None:
                labels_dic_oversampling[label] = 1
            else:
                labels_dic_oversampling[label] += 1
        # assert len(labelsDic)==2

        for k in labels_dic_oversampling:
            one_class = [k, labels_dic_oversampling[k]]
            classes_oversampling.append(one_class)
        classes_oversampling = np.array(classes_oversampling)

        for i in range(0, len(classes_oversampling)):
            print("{0: ^8}{1: ^10}{2: ^9}".format(classes_oversampling[i][0], classes_oversampling[i][1], sum(classes_oversampling[i, 1:])))
        print("{0: ^8}{1: ^10}{2: ^9}".format(" total", sum(classes_oversampling[:, 1]), classes_oversampling[:, 1:].sum()))
    else:
        dataset_resample = dataset

    vertices_features, labels, process_count = [], [], 0

    for k in dataset_resample:
        # get the label, vertices coords and features of this sample.
        [label, vertice_coords, vertice_features] = dataset_resample[k]
        # assert len(vertice_coords) == len(vertice_features)
        subobject_size = len(vertice_coords)
        new_vertice_features = [subobject_size] + vertice_features

        if subobject_size < 2:
            print("label:{} only one point".format(label))
            continue

        # # 4 collecting the sample: vertices_features, adjacency, label.
        vertices_features.append(new_vertice_features)
        labels.append(label)

    # preprocessing inputs.
    concatenate_feature = np.array(vertices_features)
    # print(concatenate_feature.shape)
    min_val = 0.00000001

    # standardizing
    if data_type == 1:
        # Calculate the mean and std of train dataset, they also will be used to validation and test dataset.
        mean_feature = concatenate_feature.mean(axis=0)
        std_feature = concatenate_feature.std(axis=0)
        std_feature += min_val

        file = "config/_config_svm.txt"
        conc = np.vstack((mean_feature, std_feature))
        np.savetxt(file, conc, fmt='%.18f')

    if data_type == -1:  # for the extra experiment.
        # Import the mean and std of train dataset.
        file = "config/_config_svn.txt"
        conc = np.loadtxt(file)
        mean_feature, std_feature = conc[0, :], conc[1, :]
        print("\n========import the mean and std of train dataset from text file========\n")
        # print(mean_feature)
        # print(std_feature)

    for i in range(0, len(vertices_features)):
        vertices_features[i] -= mean_feature
        vertices_features[i] /= std_feature


    assert len(vertices_features) == len(labels)

    return [vertices_features, labels, mean_feature, std_feature]
