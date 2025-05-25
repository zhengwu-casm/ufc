import json

def load_data(filename):
    print("Loading the {} data".format(filename))
    file = open(filename, 'r', encoding='utf-8')
    dataset = json.load(file)
    file.close()
    vertices_features, labels, process_count = [], [], 0

    for k in dataset:
        # # 1 get the label, vertices coords and features of this sample.
        [label, vertice_coords, vertice_features] = dataset[k]
        # assert len(vertice_coords) == len(vertice_features)
        subobject_size = len(vertice_coords)

        if subobject_size < 2:
            print("label:{} only one point".format(label))
            continue

        # collecting the sample: vertices_features.
        vertices_features += vertice_features
    return vertices_features
