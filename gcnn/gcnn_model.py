import tensorflow as tf
import numpy as np
import os, time, collections

import sklearn, sklearn.datasets, sklearn.utils
import sklearn.naive_bayes, sklearn.linear_model, sklearn.svm, sklearn.neighbors, sklearn.ensemble

from gcnn_layer import GraphConvolutionLayer, FullConnectionLayer
from keras import activations, regularizers, constraints, initializers

import random
random_seed = 42 #22 42 3407
random.seed(random_seed)  # set random seed for python
np.random.seed(random_seed)  # set random seed for numpy
tf.random.set_seed(random_seed)  # set random seed for tensorflow-cpu
# os.environ['TF_DETERMINISTIC_OPS'] = '1'  # set random seed for tensorflow-gpu

class gcnn(object):
    """
    GCNN model.
    input:
        Vs: Dimensions of vectice vecter (Channes).
        As: Number of vertices.          (Pixels)

    hyper-parameters:
        Fs: Number of feature maps.
        Ks: List of polynomial orders, i.e. filter sizes or number of hopes.
        Ps: Pooling size.
           Should be 1 (no pooling) or a power of 2 (reduction by 2 at each coarser level).
           Beware to have coarsened enough.
        Cs: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.

    The following are choices of implementation for various blocks.
        filter: filtering operation.
        brelu:  bias and relu.
        pool:   pooling.

    Training parameters:
        num_epochs:    Number of training epochs.
        learning_rate: Initial learning rate.
        decay_rate:    Base of exponential decay. No decay with 1.
        decay_steps:   Number of steps after which the learning rate decays.
        momentum:      Momentum. 0 indicates no momentum.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """

    def __init__(self, Vs, As, Fs, Ks, Ps, Cs, filter='monomial', brelu='b1relu', pool='m1pool',
                 num_epochs=20, batch_size=100, decay_steps=None, eval_frequency=200,
                 regularization=0, dropout=0, learning_rate=0.1, decay_rate=0.95, momentum=0.9,
                 dir_name=''):
        super().__init__()
        # Verify the consistency w.r.t. the number of layers.
        assert len(Fs) == len(Ks) == len(Ps)
        assert np.all(np.array(Ps) >= 1)
        p_log2 = np.where(np.array(Ps) > 1, np.log2(Ps), 0)
        assert np.all(np.mod(p_log2, 1) == 0)  # Powers of 2.

        # Store attributes and bind operations.
        self.Vs, self.As, self.Fs, self.Ks, self.Ps, self.Cs = Vs, As, Fs, Ks, Ps, Cs
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        # Build the computational graph.
        # Model.
        self.layers = []
        for i in range(0, len(Fs)):
            layer = GraphConvolutionLayer(self.Fs[i], filter_type=filter, brelu=brelu, pool=pool, pool_size=self.Ps[i] ,k_order=self.Ks[i])
            self.layers.append(layer)

        self.fc1 = FullConnectionLayer(self.Cs[:-1][0], userelu=True)
        self.fc2 = FullConnectionLayer(self.Cs[-1], userelu=False)
        # Learning rate.
        if decay_rate != 1:
            self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps, decay_rate, staircase=True)
        else:
            self.learning_rate = learning_rate
        # Optimizer.
        if momentum == 0:
            self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum)

        self.var_list = None

    # High-level interface which runs the constructed computational graph.
    def predict(self, vertices, adjacencies, labels):
        loss = 0
        size = vertices.shape[0]
        predictions = np.empty(size)
        # probabilities = np.empty((size, 2))
        #n_class
        # probabilities = np.empty((size, 5))
        label_class = self.Cs[1]
        probabilities = np.empty((size, self.Cs[1]))
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])

            batch_vertices = np.zeros((self.batch_size, vertices.shape[1], vertices.shape[2]))
            batch_adjacencies = np.zeros((self.batch_size, adjacencies.shape[1], adjacencies.shape[2]))
            batch_labels = np.zeros(self.batch_size)

            batch_vertices[:end - begin] = vertices[begin:end, ...]
            batch_adjacencies[:end - begin] = adjacencies[begin:end, ...]
            batch_labels[:end - begin] = labels[begin:end]

            batch_vertices = tf.convert_to_tensor(batch_vertices)
            batch_adjacencies = tf.convert_to_tensor(batch_adjacencies)

            op_logits = self.inference(batch_vertices, batch_adjacencies, self.dropout)
            batch_loss = self.loss(op_logits, batch_labels, self.regularization)
            batch_logits = self.logitsvalue(op_logits)

            loss += batch_loss

            probabilities[begin:end] = batch_logits[0][:end - begin]
            predictions[begin:end] = batch_logits[1][:end - begin]

        probabilities = np.column_stack((probabilities, predictions, labels))
        return probabilities, predictions, loss * self.batch_size / size

    # Methods to construct the computational graph.

    def inference(self, vertices, adjacencies, dropout):
        """
        It builds the model, i.e. the computational graph, as far as
        is required for running the network forward to make predictions,
        i.e. return logits given raw data.

        vertices: size N x M x
            N: number of signals (samples)
            M: number of vertices (features)
        training: we may want to discriminate the two, e.g. for dropout.
            True: the model is built for training.
            False: the model is built for evaluation.
        """
        # TODO: optimizations for sparse data
        # Graph convolutional layers.
        # vertices     N_ * M_ * F_
        # adjacencies  M_ * M_
        # N_, M_, F_, = vertices.get_shape()
        # print("x_dim_one: {0}, {1}, {2}".format(N_, M_, F_))
        for layer in self.layers:
            vertices = layer([adjacencies, vertices])
            if self.var_list is None:
                self.var_list = layer.weights
            self.var_list += layer.weights

        # Fully connected hidden layers.
        N, M, F = vertices.get_shape()
        # M, F = vertices.get_shape()
        # print("x_dim_two: {0}, {1}, {2}".format(N, M, F))
        vertices = tf.reshape(vertices, [N, int(M*F)])  # N x (M*F)
        for i, M in enumerate(self.Cs[:-1]):
            vertices = self.fc1(vertices)
            vertices = tf.nn.dropout(vertices, dropout)
            if self.var_list is None:
                self.var_list = self.fc1.weights
            self.var_list += self.fc1.weights

        # Logits linear layer, i.e. softmax without normalization.
        vertices = self.fc2(vertices)
        return vertices

    def probabilities(self, logits):
        """Return the probability of a sample to belong to each class."""
        probabilities = tf.nn.softmax(logits)
        return probabilities

    def prediction(self, logits):
        """Return the predicted classes."""
        prediction = tf.argmax(logits, axis=1)
        return prediction

    def logitsvalue(self, logits):
        """Return the logits values."""
        probabilities = tf.nn.softmax(logits)
        prediction = tf.argmax(logits, axis=1)
        return probabilities, prediction

    def loss(self, logits, labels, regularization):
        """Adds to the inference model the layers required to generate loss."""
        #cross_entropy
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        cross_entropy = tf.reduce_mean(cross_entropy)
        #regularization
        loss = cross_entropy + self.regularization * sum(map(tf.nn.l2_loss, self.var_list))
        # print(loss)
        # the weight_dacay only applys to the first layer.
        #         Same as the original implementation of GCN.
        # loss += FLAGS.weight_decay * sum(map(tf.nn.l2_loss, self.var_list))
        # loss += self.decay_rate * sum(map(tf.nn.l2_loss, self.layer1.weights))
        return loss

    def training(self, batch_vertices, batch_adjacencies, dropout, batch_labels, regularization):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        # batch_size = batch_vertices.shape[0]
        # with tf.GradientTape() as tape:
        logits = self.inference(batch_vertices, batch_adjacencies, dropout)
        # probabilities, prediction = self.logitsvalue(logits)
        _loss = self.loss(logits, batch_labels, regularization)

        # # optimize over weights
        # grad_list = tape.gradient(_loss, self.var_list)
        # # print(grad_list)
        # grads = zip(grad_list, self.var_list)
        # # print(grads)
        # self.optimizer.apply_gradients(grads)

        return _loss

    def fit(self, train_vertices, train_adjacencies, train_labels, val_vertices, val_adjacencies, val_labels):
        t_process, t_wall = time.process_time(), time.time()
        # Training.
        accuracies, accuracies_refined, losses = [], [], []
        n_train = train_vertices.shape[0]
        delta = n_train % self.batch_size
        if delta > 0:
            num_steps = int(n_train / self.batch_size) + 1
        else:
            num_steps = int(n_train / self.batch_size)
        for epoch in range(1, self.num_epochs + 1):
            step_indices = collections.deque()
            # Randomize the sequence
            # sequence = list(range(0, n_train))
            # np.random.shuffle(sequence)
            step_indices.extend(np.random.permutation(n_train))
            for step in range(1, num_steps + 1):
                # Be sure to have used all the samples before using one a second time.
                indices_length = len(step_indices)
                if indices_length < self.batch_size:
                    idx = [step_indices.popleft() for i in range(indices_length)]
                else:
                    idx = [step_indices.popleft() for i in range(self.batch_size)]

                batch_vertices, batch_adjacencies, batch_labels = train_vertices[idx, :], train_adjacencies[idx, :], \
                                                                  train_labels[idx]

                if type(batch_vertices) is not np.ndarray:
                    batch_vertices = batch_vertices.toarray()  # convert sparse matrices
                if type(batch_adjacencies) is not np.ndarray:
                    batch_adjacencies = batch_adjacencies.toarray()  # convert sparse matrices

                batch_vertices = tf.convert_to_tensor(batch_vertices)
                batch_adjacencies = tf.convert_to_tensor(batch_adjacencies)

                with tf.GradientTape() as tape:
                    loss_average = self.training(batch_vertices,
                                                 batch_adjacencies,
                                                 self.dropout,
                                                 batch_labels,
                                                 self.regularization)

                # optimize over weights
                grad_list = tape.gradient(loss_average, self.var_list)
                # print(grad_list)
                grads = zip(grad_list, self.var_list)
                # print(grads)
                self.optimizer.apply_gradients(grads)


                # Periodical evaluation of the model.
                if self.batch_size % self.eval_frequency == 0 or step == num_steps:
                    print('step {} / {} (epoch {} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                    print('  loss_average = {:.2e}'.format(loss_average))

                    string, accuracy, accuracy_refined, f1, loss, probabilities = self.evaluate(val_vertices,
                                                                                                val_adjacencies, val_labels)
                    accuracies.append(accuracy)
                    losses.append(loss)
                    accuracies_refined.append(accuracy_refined)

                    print('  validation {}'.format(string))
                    # print('  time: {:.0f}s (wall {:.0f}s)'.format(time.process_time() - t_process, time.time() - t_wall))

        print('validation accuracy: peak = {:.2f}, mean = {:.2f}'.format(max(accuracies), np.mean(accuracies[-10:])))

        t_step = (time.time() - t_wall) / num_steps
        return accuracies, accuracies_refined, losses, t_step

    def evaluate(self, vertices, adjacencies, labels):
        """
        Runs one evaluation against the full epoch of data.
        Return the precision and the number of correct predictions.
        Batch evaluation saves memory and enables this to run on smaller GPUs.

        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        labels: size N
            N: number of signals (samples)
        """
        t_process, t_wall = time.process_time(), time.time()
        probabilities, predictions, loss = self.predict(vertices, adjacencies, labels)
        # print(predictions)

        ncorrects = sum(predictions == labels)
        # print(predictions)
        accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
        f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average='weighted')
        # f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average='macro')

        ncorrects_refined = 0
        nshape = probabilities.shape
        for i in range(0, nshape[0]):
            if probabilities[i][nshape[1] - 2] == probabilities[i][nshape[1] - 1] and probabilities[i][labels[i]] > 0.8:
                ncorrects_refined = ncorrects_refined + 1
        accuracy_refined = 100 * ncorrects_refined / nshape[0]

        string = 'accuracy: {:.2f} ({:d} / {:d}), loss: {:.2e}, f1: {:.2e}'.format(accuracy, ncorrects, len(labels), loss, f1)

        return string, accuracy, accuracy_refined, f1, loss, probabilities

# gcnn model test
class gcnn_model_performance_testing(object):

    def __init__(logger):
        logger.names, logger.params = set(), {}
        logger.fit_accuracies, logger.fit_losses, logger.fit_time = {}, {}, {}
        logger.train_accuracy, logger.train_f1, logger.train_loss = {}, {}, {}
        logger.test_accuracy, logger.test_f1, logger.test_loss = {}, {}, {}
        logger.fit_accuracies_refined, logger.train_accuracy_refined, logger.test_accuracy_refined = {}, {}, {}

    def test(logger, model, name, params, train_dataset, val_dataset, test_dataset):
        logger.params[name] = params

        logger.fit_accuracies[name], logger.fit_accuracies_refined[name], logger.fit_losses[name], logger.fit_time[name] = \
            model.fit(train_dataset[0], train_dataset[1], train_dataset[2], val_dataset[0], val_dataset[1],
                      val_dataset[2])

        string, logger.train_accuracy[name], logger.train_accuracy_refined[name], logger.train_f1[name], logger.train_loss[
            name], probabilities = \
            model.evaluate(train_dataset[0], train_dataset[1], train_dataset[2])

        print('train {}'.format(string))
        string, logger.test_accuracy[name], logger.test_accuracy_refined[name], logger.test_f1[name], logger.test_loss[
            name], probabilities = \
            model.evaluate(test_dataset[0], test_dataset[1], test_dataset[2])

        print('test  {}'.format(string))

        print("Confusion martix = \n{}".format(
            sklearn.metrics.confusion_matrix(probabilities[:, -2], probabilities[:, -1])))

        logger.names.add(name)