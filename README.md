# ufc
UFC using GCNN

A GCNN-Based method for functional zone recognition by integrating building spatial morphology and courtyard-level context (Submitted to Transactions in GIS)

This project is an open source implementation of the graph convolutional neural network for classification of functional zone recognition with courtyard-level context.
The main of our work includes the following:
- A GCNN-based method for urban functional region classification was proposed.
- The method can capture the attributes of buildings within the functional region and their adjacency relationships. 
- An innovative indicator system incorporates human activities, socio-economic functions, and building geometric features was built.

Please note:
This project refers to the open source project contributed by Michaël Defferrard et al. (https://github.com/mdeff/cnn_graph) and the open source project contributed by Xiongfeng Yan et al. (https://github.com/xiongfengyan/gcnn). Thanks for their previous research work, which are important foundations for our research work. 

In this project, we adjust the framework to TensorFlow 2.0, and used ChebyNet to approximate a K-order polynomial of the Laplace matrix. We provide the data preprocessing functions and feature indicator calculation methods used in this paper. We also provided comparison methods and experimental data (with json format) to reproduce the experimental results of this article. 
It should be noted that the experimental data are calculated feature indicators, which are directly input into GCNN. Due to authorization issues, we are temporarily unable to provide the original SHP format data, but this does not affect the repetition of the experiments in this article.

You can also organize your own data and deal with the pre-processing methods provided in the project, and feed it to the GCNN model for classification.

If you have any question, please contact me: wuzheng@casm.ac.cn
