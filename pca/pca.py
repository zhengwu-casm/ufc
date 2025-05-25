import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from mpl_toolkits import mplot3d

import pca_load_dataset

# # Sample data

feature_dataset = pca_load_dataset.load_data('D:/PythonProject/function_area_identify_new/github/data/json/samples_pca.json')

feature_names = ["Area", "Peri", "LC", "AVGR", "SBRO", "LCO", "BISO",
           "WSWO", "RICC", "IPQC", "FRAC", "GIBC", "DIVC", "Elongation",
           "Ellipticity", "Concavity", "DCM", "BOT", "BOY", "CM11", "Eccentricity", "IMA", "CAR",
           "POP_W_W", "POP_W_O", "POP_W_W_R", "POP_W_O_R", "POP_R_W_R", "POP_R_O_R", "POP_WO_DIF",
           "Reside", "PublicLife", "Work"]

assert len(feature_dataset[0]) == len(feature_names)

df = pd.DataFrame.from_records(feature_dataset, columns=feature_names)
print(df)


# Calculate their correlation coefficients
pd.options.display.max_columns = len(feature_names)
# round(df.corr(), 2)

# Draw a heat map of the correlation coefficient matrix
# sns.heatmap(round(df.corr(), 2), annot=False)
# method : {'pearson', 'kendall', 'spearman'}
# sns.heatmap(round(df.corr(method="kendall"), 2), annot=False, xticklabels=1, yticklabels=1, cmap = 'RdBu')
sns.heatmap(round(df.corr(method="spearman"), 2), annot=False, xticklabels=1, yticklabels=1, cmap = 'RdBu')
plt.show()

# Data standardization
scaler = StandardScaler()
scaler.fit(df)
X = scaler.transform(df)

# Principal component pca fitting
model = PCA()
model.fit(X)
# Variance explained by each principal component
print("Variance explained by each principal component")
model.explained_variance_
print(model.explained_variance_)
# Percentage of variance explained by each principal component
print("Percentage of variance explained by each principal component")
model.explained_variance_ratio_
print(model.explained_variance_ratio_)

print("Cumulative percentage of variance explained by principal components")
print(model.explained_variance_ratio_.cumsum())
# visualization
plt.plot(model.explained_variance_ratio_, 'o-')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.title('PVE')
plt.show()

# Cumulative percentage
plt.plot(model.explained_variance_ratio_.cumsum(), 'o-')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Proportion of Variance Explained')
# plt.xlabel('principal component', fontproperties='SimHei')
# plt.ylabel('Cumulative percentage of variance explained by principal components', fontproperties='SimHei')
plt.axhline(0.9, color='k', linestyle='--', linewidth=1)
plt.title('Cumulative PVE')
# 15 principal components explains more than 90% of it.
plt.show()

# principal component kernel loading matrix (PCKM)
model.components_

columns = ['PC' + str(i) for i in range(1, len(feature_names)+1)]

pca_loadings = pd.DataFrame(model.components_, columns=df.columns, index=columns)
round(pca_loadings, 2)
print(pca_loadings)
# This matrix shows that each principal component is a linear combination of the original data,
# and the coefficients of the linear
#
# Drawing demonstration
# Visualize pca loadings

fig, ax = plt.subplots(2, 2)
plt.subplots_adjust(hspace=1, wspace=0.5)
for i in range(1, len(feature_names) + 1):
    ax = plt.subplot(6, 6, i)
    ax.plot(pca_loadings.T['PC' + str(i)], 'o-')
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(df.columns, rotation=30)
    ax.set_title('PCA Loadings for PC' + str(i))

plt.show()

# Calculate principal component scores for each sample
# PCA Scores

pca_scores = model.transform(X)
pca_scores = pd.DataFrame(pca_scores, columns=columns)
pca_scores.shape
pca_scores.head()
# Visualization of the first two principal components
# visualize pca scores via biplot

sns.scatterplot(x='PC1', y='PC2', data=pca_scores)
plt.title('Biplot')
plt.show()

# Visualization of the three principal components, three-dimensional plots
# Visualize pca scores via triplot

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_scores['PC1'], pca_scores['PC2'], pca_scores['PC3'], c='b')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.show()

# The three principal components were clustered using K-mean clustering to visualize


from sklearn.cluster import KMeans

model = KMeans(n_clusters=3, random_state=1, n_init=20)
model.fit(X)
model.labels_

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_scores['PC1'], pca_scores['PC2'], pca_scores['PC3'],
           c=model.labels_, cmap='rainbow')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()
