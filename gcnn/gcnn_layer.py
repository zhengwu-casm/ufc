# --------------------------------------------
# Define the graph convolution layer
# Functions for graph convolution layer
# --------------------------------------------
import tensorflow as tf
from keras.layers import Layer
from keras import activations, regularizers, constraints, initializers

class GraphConvolutionLayer(Layer):

    def __init__(self,
                 units,
                 activation=lambda x: x,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 filter_type='monomial',
                 brelu='b1relu',
                 pool='mpool1',
                 pool_size=None,
                 k_order=None,
                 **kwargs):

        super(GraphConvolutionLayer, self).__init__()

        # Initialization parameters
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.filter_type = filter_type
        self.k_order = k_order
        self.filter = getattr(self, filter_type)
        self.brelu = getattr(self, brelu)
        self.pool = getattr(self, pool)
        self.pool_size = pool_size

        # defines the initialization weights to be set for the Keras layer and bias, glorot_uniform
        self.kernel_initializer = initializers.get(kernel_initializer)#truncated_normal
        self.bias_initializer = initializers.get(bias_initializer)#constant

        # Regularization methods
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # Constraint: a function that imposes constraints on the weight values
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shapes):
        """ GCN has two inputs : [shape(An), shape(X)]
            input_shape:[vertex number X vertex number, vertex number X feature dimesion]
        """
        assert len(input_shapes) == 2
        features_shape = input_shapes[1]
        assert len(features_shape.dims) == 3
        graph_size = features_shape[1]  # graph size
        feature_dim = features_shape[2]  # feature dim

        # hasattr: Check if this object self has a certain attribute 'weight'
        if not hasattr(self, 'weight'):
            if self.filter_type == 'localpool':
                self.weight = self.add_weight(name="weight",
                                              shape=(feature_dim, self.units),
                                              initializer=self.kernel_initializer,
                                              regularizer=self.kernel_regularizer,
                                              constraint=self.kernel_constraint,
                                              trainable=True)
            elif self.filter_type == 'monomial':
                self.kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.1)
                self.kernel_regularizer = regularizers.L2(0.5)
                self.weight = self.add_weight(name="weight",
                                              shape=(feature_dim * self.k_order, self.units),
                                              initializer=self.kernel_initializer,
                                              regularizer=self.kernel_regularizer,
                                              constraint=self.kernel_constraint,
                                              trainable=True)
        if self.use_bias:
            if not hasattr(self, 'bias'):
                if self.filter_type == 'localpool':
                    self.bias = self.add_weight(name="bias",
                                                shape=(self.units,),
                                                initializer=self.bias_initializer,
                                                regularizer=self.bias_regularizer,
                                                constraint=self.bias_constraint,
                                                trainable=True)
                elif self.filter_type == 'monomial':
                    self.bias_initializer = tf.constant_initializer(0.1)
                    # self.bias_regularizer = regularizers.L2(0.5)
                    self.bias = self.add_weight(name="bias",
                                                shape=(self.units,),
                                                initializer=self.bias_initializer,
                                                regularizer=self.bias_regularizer,
                                                constraint=self.bias_constraint,
                                                trainable=True)
            else:
                self.bias = None

        self.built = True

        super(GraphConvolutionLayer, self).build(input_shapes)

    def call(self, inputs):
        """ GCN has two inputs : [An, X]
        """
        assert len(inputs) == 2
        adjacencies = inputs[0]
        vertices = inputs[1]
        output = None
        if self.filter_type == 'localpool':
            # isinstance function to determine if an object is of a known type
            if isinstance(vertices, tf.SparseTensor):
                h = tf.sparse.sparse_dense_matmul(vertices, self.weight)
            else:
                # The product obtained by the dot function operation between two-dimensional array matrices
                # is the matrix product
                h = tf.matmul(vertices, self.weight)
            output = tf.sparse.sparse_dense_matmul(adjacencies, h)

            if self.use_bias:
                output = tf.nn.bias_add(output, self.bias)

            if self.activation:
                output = self.activation(output)
        elif self.filter_type == 'monomial':
            vertices = self.filter(vertices, adjacencies, self.units, self.k_order)
            vertices = self.brelu(vertices)
            output = self.pool(vertices, self.pool_size)

        return output

    def monomial(self, vertices, adjacencies, Fout, K):
        # print(vertices.get_shape())
        N, M, Fin = vertices.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)

        assert N == adjacencies.shape[0]
        assert M == adjacencies.shape[1] == adjacencies.shape[2]
        X=[]
        # The efficiency can be improved by means of vectorization.
        for i in range(0, N):
            Li = adjacencies[i]                                   # M x M
            # Transform to monomial basis
            Xi_0 = vertices[i]                                    # M x Fin
            Xi   = tf.expand_dims(Xi_0, 0)                        # 1 x M x Fin
            def concat(x, x_):
                x_ = tf.expand_dims(x_, 0)                        # 1 x M x Fin
                return tf.concat([x, x_], axis=0)                 # K x M x Fin

            Xi_1 = None
            if K > 1:
                Xi_1 = tf.matmul(Li, Xi_0)
                Xi = concat(Xi, Xi_1)
            for k in range(2, K):
                Xi_2 = 2 * tf.matmul(Li, Xi_1) - Xi_0  # M x Fin*N
                Xi = concat(Xi, Xi_2)
                Xi_0, Xi_1 = Xi_1, Xi_2

            Xi = tf.reshape(Xi, [K, M, Fin])                      # K x M x Fin
            Xi = tf.transpose(Xi, [1, 2, 0])                      # M x Fin x K
            Xi = tf.reshape(Xi, [M, Fin*K])                       # M x Fin*K
            Xi = tf.matmul(Xi, self.weight)                                 # [M x Fin*K] x[Fin*k x Fout] = [M X Fout]
            X.append(Xi)
        return tf.reshape(X, [N, M, Fout])

    def b1relu(self, vertices):
        """Bias and ReLU. One bias per filter."""
        return tf.nn.relu(vertices + self.bias)

    def mpool1(self, vertices, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            # remain to be improved.
            # return vertices
            vertices = tf.expand_dims(vertices, 3)  # N x M x F x 1
            vertices = tf.nn.max_pool(vertices, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            #tf.maximum
            return tf.squeeze(vertices, [3])
        else:
            return vertices

    def apool1(self, vertices, p):
        """Average pooling of size p. Should be a power of 2."""
        if p > 1:
            vertices = tf.expand_dims(vertices, 3)  # N x M x F x 1
            vertices = tf.nn.avg_pool(vertices, ksize=[1, p, 1, 1], strides=[1, p, 1, 1], padding='SAME')
            return tf.squeeze(vertices, [3])  # N x M/p x F
        else:
            return vertices

class FullConnectionLayer(Layer):

    def __init__(self,
                 units,
                 activation=lambda x: x,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 userelu=True,
                 **kwargs):

        super(FullConnectionLayer, self).__init__()

        # Parameters for initializing training
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.userelu = userelu

        # Define the Keras layer to set initialization weights and bias, glorot_uniform
        self.kernel_initializer = initializers.get(kernel_initializer)#truncated_normal
        self.bias_initializer = initializers.get(bias_initializer)#constant

        # Regularization methods
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # Constraint: a function that imposes constraints on the weight values
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shapes):
        """ GCN has two inputs : [shape(An), shape(X)]
            input_shape:[vertex number X vertex number, vertex number X feature dimesion]
        """
        # assert len(input_shapes) == 2
        # print(input_shapes)
        feature_dim = input_shapes[1]

        # hasattr: Check if this object self has a certain attribute 'weight'
        if not hasattr(self, 'weight'):
            self.kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.1)
            self.kernel_regularizer = regularizers.L2(0.5)
            self.weight = self.add_weight(name="weight",
                                          shape=(int(feature_dim), self.units),
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint,
                                          trainable=True)

        if self.use_bias:
            if not hasattr(self, 'bias'):
                self.bias_initializer = tf.constant_initializer(0.1)
                # self.bias_regularizer = regularizers.L2(0.5)
                self.bias = self.add_weight(name="bias",
                                            shape=(self.units,),
                                            initializer=self.bias_initializer,
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint,
                                            trainable=True)
            else:
                self.bias = None

        self.built = True

        super(FullConnectionLayer, self).build(input_shapes)

    def call(self, inputs):
        """ GCN has two inputs : [An, X]
        """
        # assert len(inputs) == 1
        vertices = inputs
        vertices = tf.matmul(vertices, self.weight) + self.bias
        return tf.nn.relu(vertices) if self.userelu else vertices

