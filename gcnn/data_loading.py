# --------------------------------------------
# Load Data
# Functions for loading sample data
# --------------------------------------------

import json
import numpy as np
import graph

import scipy, sklearn, scipy.sparse.csgraph
from sklearn import metrics
from scipy.spatial import Delaunay
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
        return constructing_graph(data, False, 1)[:7]

# split dataset into train dataset, validate dataset, test dataset
def separating_dataset(dataset, data_separation):
    # Note that the input dataset should have a number of points greater than 1 in each courtyard！！！！
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
    vertices_features = []
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
        # # 1 get the category label, vertices coords and features of this sample.
        [label, vertice_coords, vertice_features] = dataset[k]
        vertices_features.append(vertice_features)
    maxnum_vertices = max([len(vertices_features[i]) for i in range(0, len(vertices_features))])

    train_data = constructing_graph(train_dic, True, 1, maxnum_vertices)

    return train_data[:5], \
           constructing_graph(val_dic, False, 2, maxnum_vertices, train_data[5], train_data[6])[:5], \
           constructing_graph(test_dic, False, 3, maxnum_vertices, train_data[5], train_data[6])[:5]


# constructing input data for graph neural networks
def constructing_graph(dataset, is_oversampling, data_type, max_dim,
                       mean_feature=0, std_feature=1, is_distance=True):
    # dataset {key,[label,pointlist]}
    if len(dataset) < 1:
        return None
    dataset_resample = {}
    if is_oversampling is True:
        # oversampling
        print("train dataset oversampling...")

        x, y = [], []
        for k in dataset:
            # # 1 get the label(category label), vertices coords and features of this sample.
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

    vertices_features, adjacencies, labels, graph_ids, process_count = [], [], [], [], 0

    for k in dataset_resample:
        # 1. get category label, vertices coords and features of this sample.
        [label, vertice_coords, vertice_features] = dataset_resample[k]
        assert len(vertice_coords) == len(vertice_features)
        subobject_size = len(vertice_coords)

        # 2. get the adjacency graph of the building group (one sample).
        # #   MST, Delaunay, K-NN
        points = np.array(vertice_coords)
        adjacency = np.zeros((subobject_size, subobject_size))
        if subobject_size < 2:
            # print("label:{} only one point".format(label))
            continue
        if subobject_size < 4:
            # print("label:{} not enough points to construct Delaunay (need 4)".format(label))
            # print("not enough points(1) to construct Delaunay (need 4)")
            # Establishment of a full connectivity matrix
            for i in range(0, subobject_size):
                for j in range(0, subobject_size):
                    if i < j:
                        adjacency[i, j] = 1
                        adjacency[j, i] = 1
        else:
            tri = Delaunay(points[:, 0:2])

            # Display the Delaunay triangular mesh constructed by the current point set
            # import matplotlib.pyplot as plt
            # plt.triplot(points[:, 0], points[:, 1], tri.simplices)
            # plt.plot(points[:, 0], points[:, 1], 'o')
            # plt.show()

            # Constructing an adjacency matrix
            for i in range(0, tri.nsimplex):
                if i > tri.neighbors[i, 2]:
                    adjacency[tri.simplices[i, 0], tri.simplices[i, 1]] = 1
                    adjacency[tri.simplices[i, 1], tri.simplices[i, 0]] = 1
                if i > tri.neighbors[i, 0]:
                    adjacency[tri.simplices[i, 1], tri.simplices[i, 2]] = 1
                    adjacency[tri.simplices[i, 2], tri.simplices[i, 1]] = 1
                if i > tri.neighbors[i, 1]:
                    adjacency[tri.simplices[i, 2], tri.simplices[i, 0]] = 1
                    adjacency[tri.simplices[i, 0], tri.simplices[i, 2]] = 1

        # 3.Calculate the weights of the adjacency matrix
        # Conversion to coo sparse matrix storage
        adjacency = scipy.sparse.coo_matrix(adjacency, shape=(subobject_size, subobject_size))
        distances = sklearn.metrics.pairwise.pairwise_distances(points[:, 0:2], metric="euclidean", n_jobs=1)

        adjacency = adjacency.multiply(distances)
        # Delaunay graph.
        # Compressed Sparse Row matrix，Return a dense ndarray representation of this sparse array
        adjacency = scipy.sparse.csr_matrix(adjacency).toarray()

        adjacency = scipy.sparse.csr_matrix(adjacency)
        assert subobject_size == points.shape[0]
        assert type(adjacency) is scipy.sparse.csr.csr_matrix

        # 4. collecting the sample: vertices_features, adjacency, label.
        vertices_features.append(vertice_features)
        adjacencies.append(adjacency)
        labels.append(label)
        graph_ids.append(k)

    # preprocessing inputs.
    min_val = 0.00000001

    # standardizing
    if data_type == 1:
        # Calculate the mean and std of train dataset, they also will be used to validation and test dataset.
        concatenate_feature = np.concatenate(vertices_features, axis=0)
        mean_feature = concatenate_feature.mean(axis=0)
        std_feature = concatenate_feature.std(axis=0)
        std_feature += min_val

        file = "config/_config_gcnn.txt"
        conc = np.vstack((mean_feature, std_feature))
        np.savetxt(file, conc, fmt='%.18f')

    if data_type == -1:  # for the extra experiment.
        # Import the mean and std of train dataset.
        file = "config/_config_gcnn.txt"
        conc = np.loadtxt(file)
        mean_feature, std_feature = conc[0, :], conc[1, :]
        # This two parameters are just for fun, do not matter.
        print("\n========import the mean and std of train dataset from text file========\n")
        # print(mean_feature)
        # print(std_feature)

    for i in range(0, len(vertices_features)):
        vertices_shape = np.array((vertices_features[i])).shape
        vertices_features[i] -= np.tile(mean_feature, vertices_shape[0]).reshape(vertices_shape)
        vertices_features[i] /= np.tile(std_feature, vertices_shape[0]).reshape(vertices_shape)

    # padding.
    # the max number of vertices in a group (sample).
    maxnum_vertices = max_dim
    graph_vertices, graph_adjacencies = [], []

    assert len(vertices_features) == len(adjacencies) == len(labels)

    for i in range(0, len(vertices_features)):
        # print(len(vertices_features[i]))
        graph_vertices.append(np.pad(vertices_features[i],
                                     ((0, maxnum_vertices - len(vertices_features[i])), (0, 0)),
                                     'constant', constant_values=(0)))
        graph_adjacencies.append(np.pad(adjacencies[i].toarray(),
                                        ((0, maxnum_vertices - adjacencies[i].shape[0]),
                                         (0, maxnum_vertices - adjacencies[i].shape[0])),
                                        'constant', constant_values=(0)))
    # collecting.
    graph_vertices = np.stack(graph_vertices, axis=0).astype(np.float32)  # NSample x NVertices x NFeature
    graph_adjacencies = np.stack(graph_adjacencies, axis=0).astype(np.float32)  # NSample x NVertices x NVertices
    graph_labels = np.array(labels).astype(np.int64)  # NSample x 1
    graph_fids = np.array(graph_ids).astype(np.int64)  # NSample x 1
    graph_size = graph_labels.shape[0]  # NSample
    graph_Laplacian = np.stack(
        [graph.laplacian(scipy.sparse.csr_matrix(A), normalized=True, rescaled=True) for A in graph_adjacencies],
        axis=0)

    return [graph_vertices,
            graph_Laplacian,
            graph_labels,
            graph_fids,
            graph_size,
            mean_feature, std_feature]
