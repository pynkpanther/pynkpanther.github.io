# Status: Draft

**THIS COURSE IS STILL IN PREPARATION**

# Differentiable Programming

**Notes:** 

* Training Neural Networks depends on Gradient Descent
* Gradient Descent depends on differentiation
* Designing network architectures would require manual calculation of partial derivatives (tedious, error prone)
* Automatic differentiation needed
* First numerical solutions? (find evidence in literature!)
* Recent solutions forward-backward paradigm? (find evidence in literature!)

## Prerequisites

* [Introduction to Machine Learning](introduction-to-ml.md) suggested but not necessary.
* [Introduction to Neural Networks](introduction-to-nns.md) suggested but not necessary.

## Exercises

* Part 1: Working with existing frameworks
* Part 2: How to implement such a framework yourself
* part 3: Best practices in deep learning

## Working with Existing Frameworks

* **TODO:** Name some frameworks here, including PyTorch, TensorFlow.

## PyTorch

The following exercises also exist in the course [Introduction to Machine Learning](introduction-to-ml.md). They teach the core concepts of neural networks, including linear regression, logistic regression, cost functions and gradient descent. While teaching these core concepts, they are also suited to get familiar with PyTorch as deep learning framework for automatic differentiation. The following PyTorch exercises are suitable for beginners with machine learnign as well as beginners with PyTorch, though if you are not familiar with the concepts mentioned, we suggest to first complete the course [Introduction to Machine Learning](introduction-to-ml.md), where you will find the same exercise and their pendants using numpy only.

* [exercise-pytorch-univariate-linear-regression](../notebooks/differentiable-programming/pytorch/exercise-pytorch-univariate-linear-regression.ipynb)
* [exercise-pytorch-multivariate-linear-regression](../notebooks/differentiable-programming/pytorch/exercise-pytorch-multivariate-linear-regression.ipynb)
* [exercise-pytorch-logistic-regression](../notebooks/differentiable-programming/pytorch/exercise-pytorch-logistic-regression.ipynb)
* [exercise-pytorch-softmax-regression](../notebooks/differentiable-programming/pytorch/exercise-pytorch-softmax-regression.ipynb)
* [exercise-pytorch-simple-neural-network](../notebooks/differentiable-programming/pytorch/exercise-pytorch-simple-neural-network.ipynb)

The exercise below is almost the same as the PyTorch exercise above but includes the implementation of activation and cost functions in numpy aswell as their derivatives and writing code to build the computational graph. This exercise is a first approach onto automatic differentation, but yet still tailored for the use case of fully connected feed forward networks only.

* [exercise-simple-neural-network](../notebooks/feed-forward-networks/exercise-simple-neural-network.ipynb)

### Tensorflow

* [exercise-tensorflow-softmax-regression.ipynb](../notebooks/differentiable-programming/tensorflow/exercise-tensorflow-softmax-regression.ipynb)


## How to Implement a Deep Learning Framework

If you ever wondered how the deep learning frameworks realize automatic differentiation, this chapter is for you.

* [exercise-automatic-differentiation-scalar](../notebooks/differentiable-programming/exercise-automatic-differentiation-scalar.ipynb)
* [exercise-automatic-differentiation-matrix](../notebooks/differentiable-programming/exercise-automatic-differentiation-matrix.ipynb)
* [exercise-automatic-differentiation-neural-network](../notebooks/differentiable-programming/exercise-automatic-differentiation-neural-network.ipynb)


## Best Practices in Deep Learning


* [exercise-weight-initialization](../notebooks/differentiable-programming/exercise-weight-initialization.ipynb)
* [exercise-activation-functions](../notebooks/differentiable-programming/exercise-activation-functions.ipynb)
* [exercise-optimizers](../notebooks/differentiable-programming/exercise-optimizers.ipynb) (SGD vs. Momentum vs. RMSProp vs. Adam)
* [exercise-dropout](../notebooks/differentiable-programming/exercise-dropout.ipynb)
* [exercise-batch-norm](../notebooks/differentiable-programming/exercise-batch-norm.ipynb)

## Documentation

[dp.py library documentation](../notebooks/differentiable-programming/addition_documentation_dp_library/html/index.html)

.
