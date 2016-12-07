import tensorflow as tf

"""
This file is meant to be a model-zoo like file that creates 
networks that can be used in other parts of the pipeline. These
networks are tensorflow references and need to expose the following API.

{ 
'state': <reference to the state placeholder>
'action': <reference to the action placeholder>
'weight': <reference to the weight placeholder>
'prob': <reference to a variable that calculates the probability of action given state>
'wlprob': <reference to a variable that calculates the weighted log probability> 
'discrete': <True/False describing whether the action space should be one-hot encoded>   
}
"""


def continuousTwoLayerReLU(sdim, adim, variance, hidden_layer=64):
    """
    This function creates a regression network that takes states and
    regresses to actions. It is based on a gated relu.

    Positional arguments:
    sdim -- int dimensionality of the state-space
    adim -- int dimensionality of the action-space
    variance -- float scaling for the probability calculation
    
    Keyword arguments:
    hidden_later -- int size of the hidden layer
    """

    x = tf.placeholder(tf.float32, shape=[None, sdim])

    a = tf.placeholder(tf.float32, shape=[None, adim])

    weight = tf.placeholder(tf.float32, shape=[None, 1])

    W_h1 = tf.Variable(tf.random_normal([sdim, hidden_layer ]))

    b_1 = tf.Variable(tf.random_normal([hidden_layer]))

    h1 = tf.concat(1,[tf.nn.relu(tf.matmul(x, W_h1) + b_1), tf.matmul(x, W_h1) + b_1])

    W_out = tf.Variable(tf.random_normal([hidden_layer*2, adim]))

    b_out = tf.Variable(tf.random_normal([adim]))

    output = tf.matmul(h1, W_out) + b_out

    logprob = tf.nn.l2_loss(output-a)

    y = tf.exp(-logprob/variance)

    wlogprob = weight*logprob
        
    return {'state': x, 
                'action': a, 
                'weight': weight,
                'prob': y, 
                'lprob': logprob,
                'wlprob': wlogprob,
                'discrete': False}



def logisticRegression(sdim, adim):

    """
    This function creates a multinomial logistic regression

    Positional arguments:
    sdim -- int dimensionality of the state-space
    adim -- int dimensionality of the action-space
    """

    x = tf.placeholder(tf.float32, shape=[None, sdim])

    a = tf.placeholder(tf.float32, shape=[None, adim])

    weight = tf.placeholder(tf.float32, shape=[None, 1])

    W_h1 = tf.Variable(tf.random_normal([sdim, adim]))
    b_1 = tf.Variable(tf.random_normal([adim]))
        
    logit = tf.matmul(x, W_h1) + b_1
    y = tf.nn.softmax(logit)

    logprob = tf.nn.softmax_cross_entropy_with_logits(logit, a)

    wlogprob = tf.transpose(tf.transpose(weight)*logprob)
        
    return {'state': x, 
                'action': a, 
                'weight': weight,
                'prob': y, 
                'lprob': logprob,
                'wlprob': wlogprob,
                'discrete': True}


def multiLayerPerceptron(sdim, 
                         adim, 
                         hidden_layer=64):
    """
    This function creates a classification network that takes states and
    predicts a hot-one encoded action. It is based on a MLP.

    Positional arguments:
    sdim -- int dimensionality of the state-space
    adim -- int dimensionality of the action-space
    
    Keyword arguments:
    hidden_later -- int size of the hidden layer
    """

    x = tf.placeholder(tf.float32, shape=[None, sdim])

    #must be one-hot encoded
    a = tf.placeholder(tf.float32, shape=[None, adim])

    #must be a scalar
    weight = tf.placeholder(tf.float32, shape=[None, 1])

    W_h1 = tf.Variable(tf.random_normal([sdim, hidden_layer]))
    b_1 = tf.Variable(tf.random_normal([hidden_layer]))
    h1 = tf.nn.sigmoid(tf.matmul(x, W_h1) + b_1)

    W_out = tf.Variable(tf.random_normal([hidden_layer, adim]))
    b_out = tf.Variable(tf.random_normal([adim]))
        
    logit = tf.matmul(h1, W_out) + b_out
    y = tf.nn.softmax(logit)

    logprob = tf.nn.softmax_cross_entropy_with_logits(logit, a)

    wlogprob = tf.transpose(tf.transpose(weight)*logprob)
        
    return {'state': x, 
                'action': a, 
                'weight': weight,
                'prob': y, 
                'lprob': logprob,
                'wlprob': wlogprob,
                'discrete': True}


def gaussianMean(sdim, adim, variance, scale):

        """
        This function creates creates a gaussian ellipsoid centered
        at a mean

        Positional arguments:
        sdim -- int dimensionality of the state-space
        adim -- int dimensionality of the action-space
        variance -- float scaling for the probability calculation
        scale -- float the scaling for the initialization

        """

        if adim != 2:
            raise ValueError("Gaussian only apply to binary")

        x = tf.placeholder(tf.float32, shape=[None, sdim])

        #must be one-hot encoded
        a = tf.placeholder(tf.float32, shape=[None, 2])

        #must be a scalar
        weight = tf.placeholder(tf.float32, shape=[None, 1])
        
        mu = scale*tf.Variable(tf.random_uniform([1,sdim]))
        N= tf.shape(x)
        MU = tf.tile(mu, [N[0],1])

        y = [1-tf.exp(-tf.reduce_sum( tf.abs(x-MU), 1)/variance), tf.exp(-tf.reduce_sum( tf.abs(x-MU), 1)/variance)]

        logprob = tf.nn.softmax_cross_entropy_with_logits(tf.transpose(y), a)

        wlogprob = weight*logprob
        
        return {'state': x, 
                'action': a, 
                'weight': weight,
                'mu':mu,
                'prob': y, 
                'lprob': logprob,
                'wlprob': wlogprob,
                'discrete': True}

def affine(sdim, adim, variance):
    """
    This function creates a linear regression network that takes states and
    regresses to actions. It is based on a gated relu.

    Positional arguments:
    sdim -- int dimensionality of the state-space
    adim -- int dimensionality of the action-space
    variance -- float scaling for the probability calculation
    
    """

    x = tf.placeholder(tf.float32, shape=[None, sdim])

    a = tf.placeholder(tf.float32, shape=[None, adim])

    weight = tf.placeholder(tf.float32, shape=[None, 1])

    W_h1 = tf.Variable(tf.random_normal([adim, sdim]))

    b_1 = tf.Variable(tf.random_normal([adim]))

    output = tf.matmul(x, W_h1) + b_1

    logprob = tf.nn.l2_loss(output-a)

    y = tf.exp(-logprob/variance)

    wlogprob = weight*logprob
        
    return {'state': x, 
                'action': a, 
                'weight': weight,
                'prob': y, 
                'lprob': logprob,
                'wlprob': wlogprob,
                'discrete': False}
    