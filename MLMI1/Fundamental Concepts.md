### Sum Rule

The sum rule, also known as the marginalization rule, allows us to compute the marginal probability of a random variable by summing (or integrating) over all possible values of another variable:

For discrete variables:
$$p(x) = \sum_y p(x, y)$$

For continuous variables:
$$p(x) = \int p(x, y) \, dy$$

### Product Rule

The product rule expresses the joint probability of two events as the product of a conditional probability and a marginal probability:

$$p(x, y) = p(x) \, p(y | x) = p(y) \, p(x | y)$$

These two rules form the basis for all probabilistic reasoning and inference.

## Bayes' Theorem

Bayes' theorem is derived from the product rule and provides a way to update our beliefs about the parameters or hypotheses in light of new data:

$$p(\theta | D) = \frac{p(D | \theta) \, p(\theta)}{p(D)}$$

- **Posterior**: $p(\theta \mid D)$: Probability of parameters $\theta$ given data $D$.
- **Likelihood**: $p(D \mid \theta)$: Probability of data $D$ given parameters $\theta$.
- **Prior**: $p(\theta)$: Initial probability of parameters $\theta$.
- **Marginal Likelihood**: $p(D)$: Probability of data $D$.
