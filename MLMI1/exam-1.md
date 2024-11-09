# MACHINE LEARNING AND MACHINE INTELLIGENCE MPhil

**MLMI 1**
---

### Question 1

1. (a) Define the concept of a conjugate prior in Bayesian statistics. Provide an example involving a likelihood function and its conjugate prior, and explain why the conjugate prior simplifies the computation of the posterior distribution. [20%]
   
   (b) Consider a dataset of $N$ independent Bernoulli trials, where each trial results in a success (1) or failure (0). Let $\theta$ be the probability of success in each trial. Assume a Beta prior distribution for $\theta$: $p(\theta)=\text{Beta}(\theta;\alpha,\beta)$, where $\alpha$ and $\beta$ are known positive constants.
   
   (i) Derive the posterior distribution $p(\theta\mid\text{data})$ after observing the data. Show that the posterior is also a Beta distribution, and specify its parameters. [40%]

   (ii) Compute the predictive distribution for the next trial, i.e., compute $p(x_{N+1}=1\mid\text{data})$. [20%]

   (c) Suppose now that the prior parameters $\alpha$ and $\beta$ are themselves random variables with Gamma hyperpriors: $\alpha\sim\text{Gamma}(a_0,b_0)$, $\beta\sim\text{Gamma}(c_0,d_0)$. Describe qualitatively how this hierarchical Bayesian model affects the inference of $\theta$, and discuss the potential computational challenges in this setting. [20%]

---

### Question 2

A data scientist is analyzing a dataset where the observations $x_n$ are assumed to be drawn independently from a Gaussian distribution with unknown mean $\mu$ and variance $\sigma^2$. However, the variance $\sigma^2$ is not known and is believed to vary from one observation to another according to an inverse gamma distribution. Specifically, for each observation $n$, the variance $\sigma_n^2$ is a random variable drawn from an inverse gamma distribution with parameters $\alpha$ and $\beta$.

1. (a) Write down the hierarchical probabilistic model that describes the joint distribution of the observations $x_n$, the variances $\sigma_n^2$, and the mean $\mu$, including any necessary prior distributions. [30%]

   (b) Derive the conditional posterior distribution $p(\mu\mid\{x_n\},\{\sigma_n^2\})$. [20%]

   (c) Propose an algorithm based on Gibbs sampling to perform inference in this hierarchical model, specifying the conditional distributions needed for the sampling steps. [50%]

---

### Question 3

Consider a mixture of Gaussians model for clustering a dataset of $D$-dimensional observations $\{x_n\}$ where $n=1,\dots,N$. The mixture model has $K$ components with mixing proportions $\pi_k$, means $\mu_k$, and shared covariance matrix $\Sigma$.

1. (a) Explain why using a shared covariance matrix across all mixture components might be advantageous in high-dimensional settings. Discuss any potential drawbacks. [20%]

   (b) Derive the complete data log-likelihood for this model, including the latent variables indicating the component memberships. [20%]

   (c) Derive the Expectation-Maximization (EM) algorithm updates for the parameters $\pi_k$, $\mu_k$, and $\Sigma$ in this mixture model. Show all steps clearly. [60%]

---

### Question 4

A Hidden Markov Model (HMM) is used to model sequences of observations $y_1,\dots,y_T$. The model has $K$ hidden states $s_t\in\{1,\dots,K\}$, and the observation model is a Gaussian distribution whose mean depends linearly on the hidden state: $y_t\mid s_t=k\sim N(y_t;w_k^\top x_t,\sigma^2)$, where $x_t$ is a known feature vector at time $t$.

1. (a) Write down the joint probability distribution of the observed sequence $\{y_t\}$ and hidden states $\{s_t\}$, specifying all model parameters. [20%]

   (b) Explain how the forward-backward algorithm can be used to compute the marginal posterior distributions $p(s_t\mid y_{1:T})$ for $t=1,\dots,T$. [30%]

   (c) Suppose you wish to learn the parameters $w_k$ and $\sigma^2$ from data. Outline how the EM algorithm can be adapted to estimate these parameters in the HMM. Include expressions for the E-step and M-step updates. [50%]

---

### Question 5

Consider a Bayesian linear regression model with a conjugate Gaussian prior on the weights $w$ and a known noise variance $\sigma^2$. Suppose that after observing $N$ data points, the posterior distribution over the weights is $p(w\mid D_N)=N(w;\mu_N,\Sigma_N)$. A new data point $(x_{N+1},y_{N+1})$ is observed.

1. (a) Show how to update the posterior distribution $p(w\mid D_{N+1})$ in closed form after observing the new data point, without recomputing the posterior from scratch. [40%]

   (b) Discuss how this sequential updating can be used in online learning scenarios, and any potential computational benefits or drawbacks compared to batch learning. [30%]

   (c) If the prior over the weights $w$ is instead a Laplace distribution, discuss qualitatively how the posterior updating would differ, and what impact this might have on the weight estimates. [30%]

---

### Question 6

The Kullback-Leibler (KL) divergence between two multivariate Gaussian distributions $p(x)=N(x;\mu_p,\Sigma_p)$ and $q(x)=N(x;\mu_q,\Sigma_q)$ is given by:

$$
\text{KL}(q\parallel p) = \frac{1}{2} \left[ \log \left( \frac{\det \Sigma_p}{\det \Sigma_q} \right) - D + \text{tr}(\Sigma_p^{-1} \Sigma_q) + (\mu_p - \mu_q)^\top \Sigma_p^{-1} (\mu_p - \mu_q) \right]
$$

1. (a) Derive this expression for the KL divergence between two multivariate Gaussians. [50%]

   (b) Given two multivariate Gaussians with identical covariances ($\Sigma_p=\Sigma_q$), simplify the KL divergence expression and interpret the result. [20%]

   (c) Explain how the KL divergence can be used in variational inference to approximate intractable posterior distributions, and discuss any limitations of this approach when dealing with multimodal distributions. [30%]

---

### Question 7

In reinforcement learning, policy gradient methods are used to optimize parameterized policies $\pi_\theta(a\mid s)$ with respect to expected cumulative reward. Consider a simple environment where an agent must choose actions $a_t\in\{0,1\}$ in states $s_t\in\{0,1\}$, and the policy is parameterized as $\pi_\theta(a_t\mid s_t)=\text{softmax}(\theta_{s_t},a_t)$.

1. (a) Derive the policy gradient $\nabla_\theta J(\theta)$ for this setup, where $J(\theta)$ is the expected cumulative reward. [40%]

   (b) Explain how the REINFORCE algorithm uses Monte Carlo sampling to estimate the policy gradient, and discuss any variance reduction techniques that can be applied. [30%]

   (c) Discuss the potential challenges of using policy gradient methods in high-dimensional action spaces, and propose possible solutions to mitigate these challenges. [30%]

---

### Question 8

A data scientist is working with a non-linear regression problem where the outputs $y_n$ are related to the inputs $x_n$ via a function $y_n=f(x_n)+\epsilon_n$, with $\epsilon_n\sim N(0,\sigma^2)$. The function $f(x)$ is assumed to be a draw from a Gaussian process (GP) with zero mean and covariance function $k(x,x')$.

1. (a) Explain what a Gaussian process is, and how it defines a distribution over functions. [20%]

   (b) Derive the posterior predictive distribution $p(y^*\mid x^*,X,y)$ for a new input $x^*$, given observed data $(X,y)$. [40%]

   (c) Discuss how the choice of covariance function $k(x,x')$ affects the properties of the GP, and how hyperparameters of $k$ can be learned from data. [40%]

---

### Question 9

Consider a logistic regression model with $L_2$ regularization, where the weights $w$ are learned by minimizing the regularized negative log-likelihood:

$$
L(w) = -\sum_{n=1}^{N} \left[ y_n \log \sigma(w^\top x_n) + (1 - y_n) \log (1 - \sigma(w^\top x_n)) \right] + \lambda \|w\|^2
$$

1. (a) Derive the gradient $\nabla_w L(w)$ of the regularized loss function. [30%]

   (b) Explain how Newton's method can be used to optimize $L(w)$, and derive the expression for the Hessian matrix $H(w)$. [30%]

   (c) Discuss the advantages and disadvantages of using Newton's method versus stochastic gradient descent for optimizing logistic regression models, particularly in large-scale settings. [40%]

---

### Question 10

In variational autoencoders (VAEs), we approximate the intractable posterior distribution $p(z\mid x)$ with a variational distribution $q(z\mid x;\phi)$, parameterized by an encoder network.

1. (a) Explain the evidence lower bound (ELBO) used in VAEs, and derive the expression for the ELBO in terms of the reconstruction error and the KL divergence between $q(z\mid x;\phi)$ and the prior $p(z)$. [30%]

   (b) Suppose we use a Gaussian prior $p(z)=N(z;0,I)$ and approximate posterior $q(z\mid x;\phi)=N(z;\mu(x;\phi),\text{diag}(\sigma^2(x;\phi)))$. Derive the KL divergence term in the ELBO for this case. [40%]

   (c) Discuss how the reparameterization trick is used to enable backpropagation through the sampling process in VAEs, and any limitations this might have for different types of variational distributions. [30%]

---

**END OF PAPER**
