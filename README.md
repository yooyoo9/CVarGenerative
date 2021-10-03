# Risk-Averse Generative Modeling

## Dataset
We use the [MNIST](http://yann.lecun.com/exdb/mnist/), the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset and the [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.
The code expects both datasets to be in the following directories:
- `experiments/mnist/input`
- `experiments/celeba/input`
- `experiments/vae/input`

## Description
The code implements three different algorithms:
- CVaR-EM: the risk-averse version of the Expectation-Maximization algorithm for Gaussian Mixture Models
- CVaR-VAE: risk-averse Variational Auto-Encoder, optimizing the CVaR of the ELBO
- CVaR-GAN: risk-averse Generative Adversarial Networks

Here are some sample commands to run the different algorithms:
- `python3 algorithms/gmm/main.py --reproduce`
- `python3 algorithms/gan/main.py --save_model`
- `python3 algorithms/vae/main.py --save_model`