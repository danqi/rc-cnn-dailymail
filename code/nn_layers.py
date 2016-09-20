import theano.tensor as T
import lasagne


def stack_rnn(l_emb, l_mask, num_layers, num_units,
              grad_clipping=10, dropout_rate=0.,
              bidir=True,
              only_return_final=False,
              name='',
              rnn_layer=lasagne.layers.LSTMLayer):
    """
        Stack multiple RNN layers.
    """

    def _rnn(backwards=True, name=''):
        network = l_emb
        for layer in range(num_layers):
            if dropout_rate > 0:
                network = lasagne.layers.DropoutLayer(network, p=dropout_rate)
            c_only_return_final = only_return_final and (layer == num_layers - 1)
            network = rnn_layer(network, num_units,
                                grad_clipping=grad_clipping,
                                mask_input=l_mask,
                                only_return_final=c_only_return_final,
                                backwards=backwards,
                                name=name + '_layer' + str(layer + 1))
        return network

    network = _rnn(True, name)
    if bidir:
        network = lasagne.layers.ConcatLayer([network, _rnn(False, name + '_back')], axis=-1)
    return network


class AveragePoolingLayer(lasagne.layers.MergeLayer):
    """
        Average pooling.
        incoming: batch x len x h
    """
    def __init__(self, incoming, mask_input=None, **kwargs):
        incomings = [incoming]
        if mask_input is not None:
            incomings.append(mask_input)
        super(AveragePoolingLayer, self).__init__(incomings, **kwargs)
        if len(self.input_shapes[0]) != 3:
            raise ValueError('the shape of incoming must be a 3-element tuple')

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:-2] + input_shapes[0][-1:]

    def get_output_for(self, inputs, **kwargs):
        if len(inputs) == 1:
            # mask_input is None
            return T.mean(inputs[0], axis=1)
        else:
            # inputs[0]: batch x len x h
            # inputs[1] = mask_input: batch x len
            return (T.sum(inputs[0] * inputs[1].dimshuffle(0, 1, 'x'), axis=1) /
                    T.sum(inputs[1], axis=1).dimshuffle(0, 'x'))


class MLPAttentionLayer(lasagne.layers.MergeLayer):
    """
        An MLP attention layer.
        incomings[0]: batch x len x h
        incomings[1]: batch x h
        Reference: http://arxiv.org/abs/1506.03340
    """
    def __init__(self, incomings, num_units,
                 nonlinearity=lasagne.nonlinearities.tanh,
                 mask_input=None,
                 init=lasagne.init.Uniform(), **kwargs):
        if len(incomings) != 2:
            raise NotImplementedError
        if mask_input is not None:
            incomings.append(mask_input)
        super(MLPAttentionLayer, self).__init__(incomings, **kwargs)
        self.nonlinearity = nonlinearity
        self.num_units = num_units
        self.W0 = self.add_param(init, (self.num_units, self.num_units), name='W0_mlp')
        self.W1 = self.add_param(init, (self.num_units, self.num_units), name='W1_mlp')
        self.Wb = self.add_param(init, (self.num_units, ), name='Wb_mlp')

    def get_output_shape_for(self, input_shapes):
        return input_shapes[1]

    def get_output_for(self, inputs, **kwargs):
        M = T.dot(inputs[0], self.W0) + T.dot(inputs[1], self.W1).dimshuffle(0, 'x', 1)
        M = self.nonlinearity(M)
        alpha = T.nnet.softmax(T.dot(M, self.Wb))
        if len(inputs) == 3:
            alpha = alpha * inputs[2]
            alpha = alpha / alpha.sum(axis=1).reshape((alpha.shape[0], 1))
        return T.sum(inputs[0] * alpha.dimshuffle(0, 1, 'x'), axis=1)


class BilinearAttentionLayer(lasagne.layers.MergeLayer):
    """
        A bilinear attention layer.
        incomings[0]: batch x len x h
        incomings[1]: batch x h
    """
    def __init__(self, incomings, num_units,
                 mask_input=None,
                 init=lasagne.init.Uniform(), **kwargs):
        if len(incomings) != 2:
            raise NotImplementedError
        if mask_input is not None:
            incomings.append(mask_input)
        super(BilinearAttentionLayer, self).__init__(incomings, **kwargs)
        self.num_units = num_units
        self.W = self.add_param(init, (self.num_units, self.num_units), name='W_bilinear')

    def get_output_shape_for(self, input_shapes):
        return input_shapes[1]

    def get_output_for(self, inputs, **kwargs):

        # inputs[0]: batch * len * h
        # inputs[1]: batch * h
        # W: h * h

        M = T.dot(inputs[1], self.W).dimshuffle(0, 'x', 1)
        alpha = T.nnet.softmax(T.sum(inputs[0] * M, axis=2))
        if len(inputs) == 3:
            alpha = alpha * inputs[2]
            alpha = alpha / alpha.sum(axis=1).reshape((alpha.shape[0], 1))
        return T.sum(inputs[0] * alpha.dimshuffle(0, 1, 'x'), axis=1)


class DotProductAttentionLayer(lasagne.layers.MergeLayer):
    """
        A bilinear attention layer.
        incomings[0]: batch x len x h
        incomings[1]: batch x h
    """
    def __init__(self, incomings, mask_input=None, **kwargs):
        if len(incomings) != 2:
            raise NotImplementedError
        if mask_input is not None:
            incomings.append(mask_input)
        super(DotProductAttentionLayer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[1]

    def get_output_for(self, inputs, **kwargs):

        # inputs[0]: batch * len * h
        # inputs[1]: batch * h
        # mask_input (if any): batch * len

        alpha = T.nnet.softmax(T.sum(inputs[0] * inputs[1].dimshuffle(0, 'x', 1), axis=2))
        if len(inputs) == 3:
            alpha = alpha * inputs[2]
            alpha = alpha / alpha.sum(axis=1).reshape((alpha.shape[0], 1))
        return T.sum(inputs[0] * alpha.dimshuffle(0, 1, 'x'), axis=1)
