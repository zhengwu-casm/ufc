# -*- coding: utf-8 -*-
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import rf_load_dataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Preparing training samples
add_building_count = False

train_dataset, val_dataset, test_dataset = rf_load_dataset.load_data('D:/PythonProject/function_area_identify_new/github/data/json/samples_rf.json',
                                                                  add_building_count, data_separation=[0.6, 0.2, 0.2])

[train_x, train_y, mean_train_x, std_train_x] = train_dataset
[val_x, val_y, mean_val_x, std_val_x] = val_dataset
[test_x, test_y, mean_test_x, std_test_x] = test_dataset

# 1.Decision tree
clf1 = DecisionTreeClassifier(random_state=0)
clf1.fit(train_x, train_y)
# Predicting results on a test set
predicted_1_y = clf1.predict(test_x)
scores1 = clf1.score(test_x, test_y)
# print(scores1)
# scores1_ = cross_val_score(clf1, train_x, train_y)
# print(scores1_.mean())
# Print prediction results and model scores
print("DecisionTree: ")
print("Predicted labels: ", predicted_1_y)
print("Accuracy score: ", scores1)

# Output evaluation indicators
print("Accuracy:", accuracy_score(test_y, predicted_1_y))
print("Precision:", precision_score(test_y, predicted_1_y, average='macro'))
print("Recall:", recall_score(test_y, predicted_1_y, average='macro'))
print("F1 Score:", f1_score(test_y, predicted_1_y, average='macro'))
print("Confusion Matrix:")
print(confusion_matrix(test_y, predicted_1_y))
print("#####################################")

# 2.random forest
clf2 = RandomForestClassifier(random_state=0)
clf2.fit(train_x, train_y)
# Predicting results on a test set
predicted_2_y = clf2.predict(test_x)
scores2 = clf2.score(test_x, test_y)
# print(scores2)
# scores2_ = cross_val_score(clf2, train_x, train_y)
# print(scores2_.mean())
print("RandomForest: ")
print("Predicted labels: ", predicted_2_y)
print("Accuracy score: ", scores2)

# Output evaluation indicators
print("Accuracy:", accuracy_score(test_y, predicted_2_y))
print("Precision:", precision_score(test_y, predicted_2_y, average='macro'))
print("Recall:", recall_score(test_y, predicted_2_y, average='macro'))
print("F1 Score:", f1_score(test_y, predicted_2_y, average='macro'))
print("Confusion Matrix:")
print(confusion_matrix(test_y, predicted_2_y))
print("#####################################")

dt_importances = clf1.feature_importances_
print("DT importance:", dt_importances)

rf_importances = clf2.feature_importances_
print("RF importance:", rf_importances)