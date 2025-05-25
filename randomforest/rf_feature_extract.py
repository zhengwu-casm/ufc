import numpy as np
import json
import csv

# Extraction Characteristics of sample data
def extract_features_json(result_file, output_file, fields_list):
    print("Loading the {} data".format(result_file))
    file = open(result_file, 'r', encoding='utf-8')
    dataset = json.load(file)
    file.close()

    # fields_list = ["long_chord", "avg_radius", \
    #                "ric_compa", "concavity", "m11", \
    #                "eccentric", "pop_w_d_r", "pop_w_n_r",\
    #                "pop_r_d_r", "pop_r_n_r", "pop_wr_dif", "reside", "life", "company", "public", "tour"]

    label_feature_dic = {}
    for k in dataset:
        # # 1 get the label, vertices coords and features of this sample.
        [label, vertice_coords, vertice_features] = dataset[k]
        yard_features = []
        vertice_features_array = np.array(vertice_features)
        dim = vertice_features_array.shape[1]
        # yard_features.append(len(vertice_coords))
        for i in range(0, dim):
            min = vertice_features_array.min(0)
            max = vertice_features_array.max(0)
            mean = vertice_features_array.mean(0)
            # variance
            var = vertice_features_array.var(0)
            # var = np.var(vertice_features_array)
            # standard deviation
            std = vertice_features_array.std(0)
            val = fields_list[i]
            if val == "min":
                yard_features.append(min[i])
            elif val == "max":
                yard_features.append(max[i])
            elif val == "mean":
                yard_features.append(mean[i])
            else:
                yard_features.append(std[i])
        label_feature_dic[k] = [label, vertice_coords, yard_features]

    with open(output_file, 'w') as json_file:
        json.dump(label_feature_dic, json_file, indent=2, ensure_ascii=False)

    return


def test_extract_features_json():
    result_file = "D:/PythonProject/function_area_identify_new/github/data/json/samples.json"
    output_file = "D:/PythonProject/function_area_identify_new/github/data/json/samples_rf.json"

    fields_list = ["mean", "mean", "mean", "mean", "mean", "mean", "mean", "mean", "mean", "mean", \
                   "mean", "mean", "mean", "mean", "mean", "mean", "mean", "mean", "mean", "mean", \
                   "mean", "mean", "mean", \
                   "mean", "mean", "mean", \
                   "mean", "mean", "mean", "mean", "mean", "mean", "mean"]

    # fields_list = ["std", "std", "std", "std", "std", "std", "std", \
    #                "std", "std", "std", "std", \
    #                "mean", "mean", "mean","mean", "mean", \
    #                "mean", "mean", "mean"]
    extract_features_json(result_file, output_file, fields_list)
    return


if __name__ == '__main__':
    test_extract_features_json()
