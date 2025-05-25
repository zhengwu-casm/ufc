from sklearn import svm
import numpy as np
import svm_load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Preparing training samples

train_dataset, val_dataset, test_dataset = svm_load_dataset.load_data('D:/PythonProject/function_area_identify_new/github/data/json/samples_svm.json',
                                                                  data_separation=[0.6, 0.2, 0.2])
'''
    Description:
    Kernel functions (here is a brief introduction to the four kernel functions of svm in sklearn)

    LinearSVC: mainly used in the case of linearly differentiable. 
    RBF: mainly used for linearly indivisible cases. 
    polynomial: polynomial function, degree indicates the degree of polynomial ----- supports nonlinear classification.
    Sigmoid: S-growth curve.
'''

if __name__ == "__main__":
    # get the features, label, mean, variance  of this sample.
    [train_x, train_y, mean_train_x, std_train_x] = train_dataset
    [val_x, val_y, mean_val_x, std_val_x] = val_dataset
    [test_x, test_y, mean_test_x, std_test_x] = test_dataset

    # Selecting the kernel function
    for n in range(0, 4):
        if n == 0:
            # SVM Classifier model training
            svm_model = svm.SVC(kernel='linear', C=1.0)
            clf = svm_model.fit(train_x, train_y)
            # Predictive testing dataset
            predicted_y = svm_model.predict(test_x)

            # Print prediction results and model scores
            print("linear: ")
            print("Predicted labels: ", predicted_y)
            print("Accuracy score: ", svm_model.score(test_x, test_y))

            # Output evaluation indicators
            print("Accuracy:", accuracy_score(test_y, predicted_y))
            print("Precision:", precision_score(test_y, predicted_y, average='macro'))
            print("Recall:", recall_score(test_y, predicted_y, average='macro'))
            print("F1 Score:", f1_score(test_y, predicted_y, average='macro'))
            print("Confusion Matrix:")
            print(confusion_matrix(test_y, predicted_y))
            print("#####################################")
        elif n == 1:
            svm_model = svm.SVC(kernel='poly', degree=3)
            clf = svm_model.fit(train_x, train_y)
            # Predictive testing dataset
            predicted_y = svm_model.predict(test_x)

            # Print prediction results and model scores
            print("poly: ")
            print("Predicted labels: ", predicted_y)
            print("Accuracy score: ", svm_model.score(test_x, test_y))

            # Output evaluation indicators
            print("Accuracy:", accuracy_score(test_y, predicted_y))
            print("Precision:", precision_score(test_y, predicted_y, average='macro'))
            print("Recall:", recall_score(test_y, predicted_y, average='macro'))
            print("F1 Score:", f1_score(test_y, predicted_y, average='macro'))
            print("Confusion Matrix:")
            print(confusion_matrix(test_y, predicted_y))
            print("#####################################")
        elif n == 2:
            svm_model = svm.SVC(kernel='rbf')
            clf = svm_model.fit(train_x, train_y)
            # Predictive testing dataset
            predicted_y = svm_model.predict(test_x)

            # Print prediction results and model scores
            print("rbf: ")
            print("Predicted labels: ", predicted_y)
            print("Accuracy score: ", svm_model.score(test_x, test_y))

            # Output evaluation indicators
            print("Accuracy:", accuracy_score(test_y, predicted_y))
            print("Precision:", precision_score(test_y, predicted_y, average='macro'))
            print("Recall:", recall_score(test_y, predicted_y, average='macro'))
            print("F1 Score:", f1_score(test_y, predicted_y, average='macro'))
            print("Confusion Matrix:")
            print(confusion_matrix(test_y, predicted_y))
            print("#####################################")
        else:
            svm_model = svm.SVC(kernel='sigmoid')
            clf = svm_model.fit(train_x, train_y)
            # Predictive testing dataset
            predicted_y = svm_model.predict(test_x)

            # Print prediction results and model scores
            print("sigmoid: ")
            print("Predicted labels: ", predicted_y)
            print("Accuracy score: ", svm_model.score(test_x, test_y))
            # Output evaluation indicators
            print("Accuracy:", accuracy_score(test_y, predicted_y))
            print("Precision:", precision_score(test_y, predicted_y, average='macro'))
            print("Recall:", recall_score(test_y, predicted_y, average='macro'))
            print("F1 Score:", f1_score(test_y, predicted_y, average='macro'))
            print("Confusion Matrix:")
            print(confusion_matrix(test_y, predicted_y))
            print("#####################################")

