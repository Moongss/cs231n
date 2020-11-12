# -*- coding: utf-8 -*-
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights. (3073, 10)
  - X: A numpy array of shape (N, D) containing a minibatch of data. (500, 3073)
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means (500, )
    that X[i] has label c, where 0 <= c < C. 
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # Loss = max(0, s_j - s_y_i + 1)
  # compute the loss and the gradient
  num_classes = W.shape[1] #10
  num_train = X.shape[0]   #500
  loss = 0.0
  for i in xrange(num_train): # 500
    scores = X[i].dot(W) # (500, 3073) * (3073, 10) = (500, 10)
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] > if y[i] = c, c is correct score
    correct_class_score = scores[y[i]] 
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #S_j = x_1w_1 + x_2w_2 + ... x_nw_n > x_1+x_2+...+x_n (i != correct)
        #S_y = x*w_y_i. > x[i]. 
        #왜 이렇게 되는지 종이에 써보자.. 아직 arr/broadcast 개념이 약한듯.
        dW[:, y[i]] = dW[:, y[i]] - X[i] 
        dW[:, j] = dW[:, j] + X[i]
    #####################################################################
    # TO DEBUG                                                          #
    # if i <= 2:
    #   print(dW[1])
    #####################################################################

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W # same shape

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)

  #위 코드는 같았는데, 아래 93번째 코드 왜 이렇게 돌아가는지 체크.
  #코드 작성관련해서 모르는게 꽤 많은 것 같다..
  correct_class_scores = scores[ np.arange(num_train), y].reshape(num_train,1)
  margin = np.maximum(0, scores - correct_class_scores + 1)
  margin[ np.arange(num_train), y] = 0 # do not consider correct class in loss
  
  loss = margin.sum() / num_train
 
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margin[margin > 0] = 1
  valid_margin_count = margin.sum(axis=1)
  # Subtract in correct class (-s_y)
  margin[np.arange(num_train),y ] -= valid_margin_count
  dW = (X.T).dot(margin) / num_train

  dW = dW + reg * 2 * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
