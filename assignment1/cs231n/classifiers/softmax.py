# -*- coding: utf-8 -*-
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  D : 32 * 32 * 3 = 3072 + 1 (bias)
  C : 10 (maybe)
  N : 4900 Maybe
  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  # 정규화 주의, loop 돌면서 softmax_loss, grad 구하기.                       #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = X.dot(W) # XW = (num_train=4900, num_classes=10)

  for i in range(num_train):
    current_row = scores[i] #(, 10)
    total_score = np.sum(np.exp(current_row)) #e^n_1 + e^n_2 + ...e^n_class_num : scalar
    current_score = np.exp(current_row) # ([e^n_1, e^n_2, ... e^n_class_num]) (0, class_num)
    softmax = current_score / total_score # (0, 10)
    loss += -np.log(softmax[y[i]]) # y로 들어온 train 정답값을 -log취하면 loss.

    # 미분했을때.. 실제로 해보면 지수미분때문에 inputdata X * softmax 형태로 나온다. 맞나?
    # e^(x_1*w_1) / dw = x_1 * e^(x_1*w_1)이라 softmax값은 유지. 앞에 
    for j in range(num_classes):
      dW[:, j] += X[i] * softmax[j] #j번째 col에 X*softmax를 다 더해줌. ()
    dW[:, y[i]] -= X[i] #그리고 정답라벨 한해서만 데이터 빼주기. (-log(e^softmax[y_i]))
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  loss = loss / num_train
  dW = dW / num_train

  loss += reg * np.sum(W * W)
  dW += reg * 2 * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0] 
  scores = X.dot(W) # XW = (num_dev=500, num_classes=10) (N.C)
  
  sum_exps = np.sum(np.exp(scores), axis=1, keepdims=True) # push (500, 10)
  softmax = np.exp(scores) / sum_exps # (500, 10) (N, C)
  # TO TEST
  # print(softmax.shape)
  # print(np.arange(num_train))
  # print(y.shape)
  # print(softmax[np.arange(num_train), y])

  # 아래코드 ,, 이해는 하겠는데. 막상 데이터 작게 만들어서 테스트하면 shape mismatch or
  # out of bounds가 뜬다. 흠.. 다시 테스트해보자.

  # AFTER TEST
  # 위에 로그찍어봤더니.. np.arange랑 y랑 크기가 같고, y는 num_class의 범위를 가지고 있어야 한다.
  # 만약 y에 11이 있으면 out of bounds 에러가 뜸.
  loss = np.sum(-np.log(softmax[np.arange(num_train), y])) #idx : (0~4900, y[0~4900])

  # 행렬 transpose해서 dotproduct하면,, 이게 -1을 뺀게 실제 정답값과 곱해져 자연스레 빠지게 된다.
  softmax[np.arange(num_train),y] -= 1
  # 이건 걍.. softmax와 데이터 곱해주는 과정.
  dW = X.T.dot(softmax)

  loss /= num_train
  dW /= num_train

  loss += reg * np.sum(W * W)
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

