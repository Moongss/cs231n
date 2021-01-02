# -*- coding: utf-8 -*-
from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    D = np.prod(x[0].shape)             #D = (4, 5, 6) prod
    temp_x = x.reshape(x.shape[0], D)   #re(2, 120) (N, dim)
    out = temp_x.dot(w) + b             #(2, 120) * (120, 3) + (3, )
                                        #(n, dim) * (dim, m) + (m, )
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream **derivative**, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    D = np.prod(x[0].shape)
    N = x.shape[0]
    X = x.reshape(N, D)

    #gradient 계산하기 (dot product 진행 - assignment1 참고)
    dx = dout.dot(w.T)  #(N, M) * (M, D) -> (N, D)
    dx = dx.reshape(x.shape)
    dw = X.T.dot(dout)  #(D, N) * (N, M) -> (D, M)
    db = dout.sum(axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    
    #TO DEBUG
    # print(out)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    #참고 : https://ayoteralab.tistory.com/entry/ANN-10-%ED%99%9C%EC%84%B1%ED%99%94%ED%95%A8%EC%88%98s-Back-Propagation-ReLU-Sigmoid
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout
    dx[x < 0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    훈련 중에 표본 평균과 (수정되지 않은) 표본 분산은 미니 배치 통계에서 계산되고 
    들어오는 데이터를 정규화하는 데 사용됩니다. 훈련 중에 우리는 또한 각 특징의 
    평균 및 분산의 지수 적으로 감소하는 평균을 유지하며 이러한 평균은 테스트 
    시간에 데이터를 정규화하는 데 사용됩니다.

    각 시간 단계에서 모멘텀 매개 변수를 기반으로 지수 감소를 사용하여 평균 및
     분산에 대한 실행 평균을 업데이트합니다.

    배치 정규화 문서는 다른 테스트 시간을 제안합니다.
    동작 : 실행 평균을 사용하는 대신 많은 수의 훈련 이미지를 사용하여 
    각 특징에 대한 샘플 평균과 분산을 계산합니다. 이 구현에서는 추가 추정 
    단계가 필요하지 않기 때문에 실행 평균을 사용하기로 선택했습니다. 
    배치 정규화의 torch7 구현은 또한 실행 평균을 사용합니다.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################

        #https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        mean = np.mean(x, axis=0)      #mini-batch mean
        
        xmean = (x - mean)
        sq = xmean ** 2
        var = 1./N * np.sum(sq, axis=0)  #mini-batch variance
        sqrtvar = np.sqrt((var + eps))

        ivar = 1./ sqrtvar
        x_norm = xmean * ivar             #normalize
        out = gamma * x_norm + beta            #scale and shift

        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var

        cache = (mean, xmean, sq, var, sqrtvar, ivar, x_norm, out, eps, gamma)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        
        norm = (x - running_mean) / np.sqrt(running_var + eps) #below link
        out = gamma * norm + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    (mean, xmean, sq, var, sqrtvar, ivar, x_norm, out, eps, gamma) = cache

    N, D = dout.shape
    #step 9  -> (N, D)
    dgamma_and_normal_input = 1 * dout #same as dout
    dbeta = 1 * np.sum(dout, axis=0)  #to fit dimension

    # print(dgamma_and_normal_input.shape)
    # print(dbeta.shape)
    #step 8 => dgamma and normal input (N, D)
    dnormal_input = dgamma_and_normal_input * gamma
    dgamma = np.sum(dgamma_and_normal_input * x_norm, axis=0)

    # print("===")
    # print(dnormal_input.shape)
    # print(dgamma.shape)
    #step 7 -> dnormal_input (N, D)
    dxmu_1 = dnormal_input * ivar
    dinv_den = np.sum(dnormal_input * xmean, axis=0)

    # print("===")
    # print(dxmu_1.shape)
    # print(dinv_den.shape)
    #step 6 -> inv_den (1/x -> -1/x^2)
    dsqrt_var = (-1.) / (sqrtvar ** 2) * dinv_den

    # print("===")
    # print(dsqrt_var.shape)
    #step 5
    dvar = 1. / (2 * np.sqrt(var + eps)) * dsqrt_var

    # print("===")
    # print(dvar.shape)

    #step 4
    dsq = (1. / N) * np.ones((N, D)) * dvar
    # print("===")
    # print(dsq.shape)

    #step 3
    dxmu_2 = (2 * xmean) * dsq
    # print("===")
    # print(dxmu_2.shape)

    #step 2
    xmu_sum = (dxmu_1 + dxmu_2)

    dx_1 = xmu_sum
    dmu = -1 * np.sum(xmu_sum, axis=0)
    # print("===")
    # print(dx_1.shape)
    # print(dmu.shape)
    #step 1
    dx_2 = (1. / N) * np.ones((N, D)) * dmu

    # print("===")
    # print(dx_2.shape)
    #step 0
    dx = dx_1 + dx_2
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # print(dx.shape)

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    (x, eps, gamma, err, square_err, var, var_plus_eps, inverse_den, x_norm, out) = cache
    N, D = dout.shape

	  # intermediate partial derivatives
    dxhat = dout * gamma

    # final partial derivatives
    dx = (1 / N) * inverse_den * (N*dxhat - np.sum(dxhat, axis=0) - x_norm*np.sum(dxhat*x_norm, axis=0))
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(x_norm*dout, axis=0)


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None
    #https://wiseodd.github.io/techblog/2016/06/25/dropout/
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = np.random.rand(*x.shape) < p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x * p
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH. -> **why HH?**

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    stride, pad = conv_param['stride'], conv_param['pad']
    H_ = (1 + (H + 2 * pad - HH) // stride)
    W_ = (1 + (W + 2 * pad - WW) // stride)

    out = np.zeros((N, F, H_, W_))
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0) #1, 1
    
    for n in range(N):
      for f in range(F):
        for h in range(H_):
          for w_idx in range(W_):
            x_point = x_pad[n, :, h*stride:h*stride+HH, w_idx*stride:w_idx*stride+WW]
            out[n, f, h, w_idx] = np.sum(x_point * w[f]) + b[f]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    (x, w, b, conv_param) = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    
    stride = conv_param['stride']
    pad = conv_param['pad']

    _, _, H_, W_ = dout.shape
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
    
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for n in range(N):
      for f in range(F):
        db[f] += np.sum(dout[n, f]) #ch add
        for h in range(H_):
          for w_idx in range(W_):
            dw[f] += x_pad[n, :, h*stride : h*stride+HH, w_idx*stride : w_idx*stride+WW] * dout[n, f, h, w_idx]
            dx_pad[n, :, h*stride : h*stride+HH, w_idx*stride : w_idx*stride+WW] += w[f] * dout[n, f, h, w_idx]
    dx = dx_pad[:,:,1:-1,1:-1]      

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    H_INT   = int(1 + (H - pool_height) / stride)
    W_INT   = int(1 + (W - pool_width) / stride)

    out = np.zeros((N, C, H_INT, W_INT))
    switches = {}

    for n in range(N):
      for c in range(C):
        for i in range(H_INT):
          for j in range(W_INT):
            region_of_x = x[n, 
                            c,
                            i*stride : i*stride+pool_height,
                            j*stride : j*stride+pool_width]
            out[n, c, i, j] = np.max(region_of_x)
            switches[n, c, i, j] = np.unravel_index(region_of_x.argmax(), region_of_x.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param, switches)
    return out, cache


#i don't understand clearly.. 
def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param, switches = cache

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    _, _, HH, WW = dout.shape

    dx = np.zeros(x.shape)

    for n in range(N):
      for c in range(C):
        for i in range(HH):
          for j in range(WW):
             local_index_of_max = switches[n, c, i, j]
             i_of_max = local_index_of_max[0] + i*stride
             j_of_max = local_index_of_max[1] + j*stride
             dx[n, c, i_of_max, j_of_max] += dout[n, c, i, j]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = np.shape(x)

    x_flattened = x.transpose(0, 2, 3, 1).reshape(-1, C)                         # dim=(N*H*W, C)
    out_flattened, cache = batchnorm_forward(x_flattened, gamma, beta, bn_param) # dim=(N*H*W, C)
    out = out_flattened.reshape(N, H, W, C).transpose(0, 3, 1, 2)                # dim=N,C,H,W
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = np.shape(dout)

    dout_flattened = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx_flattened, dgamma, dbeta = batchnorm_backward(dout_flattened, cache)
    dx = dx_flattened.reshape(N, H, W, C).transpose(0, 3, 1, 2)   
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
