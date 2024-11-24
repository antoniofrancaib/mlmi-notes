- [33-Introduction-to-Text-Modeling](#33-Introduction-to-Text-Modeling)
- [34-Discrete-Binary-Distributions](#34-Discrete-Binary-Distributions)
- [35-Discrete-Categorical-Distribution](#35-Discrete-Categorical-Distribution)

# 33-Introduction-to-Text-Modeling

Text modeling focuses on representing and analyzing textual data using probabilistic models. Understanding text modeling is crucial for tasks such as document classification, topic modeling, and information retrieval.

## Key Concepts
- **Modeling Document Collections**: Approaches to represent and analyze large sets of documents.
- **Probabilistic Models of Text**: Statistical methods to model the generation of text data.
- **Zipf's Law**: An empirical law describing the frequency distribution of words in natural language.
- **Bag-of-Words Representations**: A simplified representation of text that disregards grammar and word order but keeps multiplicity.

## Modeling Text Documents

### Why Model Text Documents?
Text documents are rich sources of information but are inherently unstructured. Modeling text allows us to:
- **Extract Meaningful Patterns**: Identify topics, sentiments, or authorship.
- **Compress Information**: Represent documents in a lower-dimensional space.
- **Facilitate Search and Retrieval**: Improve the efficiency of information retrieval systems.
- **Enable Automatic Categorization**: Classify documents into predefined or discovered categories.

### How to Model a Text Document?
One common approach is to represent a document by the frequency of occurrence of each distinct word it contains. This is known as the bag-of-words model.

- **Bag-of-Words Model**: Represents text as an unordered collection of words, disregarding grammar and word order.
- **Word Counts**: The frequency with which each word appears in the document.

## Word Counts in Text

![[Pasted image 20241123105843.png]]

### Word Frequency Analysis
By analyzing word frequencies, we observe that for different text collection, similar behaviours:
- **High-Frequency Words**: A small number of words occur very frequently.
- **Low-Frequency Words**: A large number of words occur infrequently.

This phenomenon is described by Zipf's Law.

## Zipf's Law

Zipf's Law states that in a given corpus of natural language, the frequency of any word is inversely proportional to its rank in the frequency table.

Mathematically:
$$f(r) \propto \frac{1}{r}$$

- $f(r)$: Frequency of the word with rank $r$.
- $r$: Rank of the word when ordered by decreasing frequency.

![[Pasted image 20241123110039.png]]

- The x-axis is the cumulative fraction of distinct words included (from the most frequent to the least frequent).
- The y-axis is the **frequency of the current word** at that point in the ranking.
### Observations from the Datasets
- **Frequency Distribution**: When plotting word frequencies against their ranks on a log-log scale, we obtain a roughly straight line, indicating a power-law distribution.
- **Cumulative Fraction**: A small fraction of the most frequent words accounts for a significant portion of the total word count.

## Automatic Categorization of Documents

**Goal**: Develop an unsupervised learning system to automatically categorize documents based on their content without prior knowledge of categories.

### Challenges
- **Unsupervised Learning**: No labeled data to guide the categorization.
- **Unknown Categories**: The system must discover categories from the data.
- **Definition of Categories**: Need to define what it means for a document to belong to a category.

### Approaches
- **Clustering**: Group documents based on similarity in word distributions.
- **Topic Modeling**: Use probabilistic models to discover latent topics (e.g., Latent Dirichlet Allocation).

--- 
# 34-Discrete-Binary-Distributions

Discrete binary distributions are fundamental in modeling binary outcomes, such as coin tosses or binary features in text data.

### Coin Tossing
- **Question**: Given a coin, what is the probability $p$ of getting heads?
- **Challenge**: Estimating $p$ based on observed data (coin toss outcomes).

#### Maximum Likelihood Estimation (MLE)
- **Single Observation**: If we observe one head ($H$), MLE suggests $p=1$.
- **Limitations**: With limited data, MLE can give extreme estimates.

#### Need for More Data
- **Additional Observations**: Suppose we observe $HHTH$.
- **MLE Estimate**: $p=\frac{3}{4}$.
- **Intuition**: Estimates become more reliable with more data.

### Bernoulli Distribution

The Bernoulli distribution models a single binary trial.

- **Random Variable**: $X \in \{0,1\}$.
  - $X=1$ represents success (e.g., heads).
  - $X=0$ represents failure (e.g., tails).
- **Parameter**: $p$, the probability of success.

#### Probability Mass Function (PMF)
$$P(X=x \mid p) = p^x (1-p)^{1-x}$$

- For $x=1$: $P(X=1 \mid p) = p$.
- For $x=0$: $P(X=0 \mid p) = 1-p$.

#### Maximum Likelihood Estimation
Given data $D = \{x_1, x_2, \dots, x_n\}$:

- **Likelihood Function**:
  $$L(p) = \prod_{i=1}^{n} p^{x_i} (1-p)^{1-x_i}$$

- **Log-Likelihood**:
  $$\ell(p) = \sum_{i=1}^{n} \left[x_i \log p + (1-x_i) \log (1-p)\right]$$

- **MLE Estimate**:
  $$\hat{p}_{\text{MLE}} = \frac{\sum_{i=1}^{n} x_i}{n}$$

### Binomial Distribution

#### Definition
The binomial distribution models the number of successes $k$ in $n$ independent Bernoulli trials.

- **Parameters**:
  - $n$: Number of trials.
  - $p$: Probability of success in each trial.

#### Probability Mass Function
$$P(k \mid n, p) = \binom{n}{k} p^k (1-p)^{n-k}$$

where the **Binomial Coefficient**  $\binom{n}{k} = \frac{n!}{k!(n-k)!}$

#### Interpretation
- **Order Independence**: The binomial distribution considers all possible sequences with $k$ successes equally likely.
- **Use Case**: When only counts of successes are important, not the specific sequence.

#### Naming of discrete distributions
![[Pasted image 20241123111817.png]]

### Bayesian Inference and Priors

#### Limitations of MLE
- **Overconfidence**: MLE can give extreme estimates with limited data.
- **No Incorporation of Prior Knowledge**: MLE relies solely on observed data.

#### Bayesian Approach
- **Incorporate Prior Beliefs**: Use prior distributions to represent initial beliefs about parameters.
- **Update Beliefs with Data**: Compute the posterior distribution using Bayes' theorem.

#### Priors and Pseudo-Counts
- **Pseudo-Counts**: Represent prior beliefs as if we have observed additional data.
  - E.g., believing the coin is fair corresponds to pseudo-counts of $\alpha=\beta=1$.
- **Strength of Belief**: Larger pseudo-counts ($\alpha=\beta=1000$) indicate stronger prior beliefs.

### Beta Distribution

#### Definition
The Beta distribution is a continuous probability distribution defined on the interval $[0, 1]$, suitable for modeling probabilities.

- **Parameters**: Shape parameters $\alpha > 0$ and $\beta > 0$.

- **Probability Density Function (PDF)**:
  $$\text{Beta}(p \mid \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} p^{\alpha-1} (1-p)^{\beta-1}$$

  where $\Gamma(\cdot)$ is the gamma function, which generalizes the factorial function.

![[Pasted image 20241123112145.png]]

#### Properties
- **Mean**:
  $$E[p] = \frac{\alpha}{\alpha + \beta}$$
- **Conjugate Prior**: The Beta distribution is conjugate to the Bernoulli and binomial distributions.

### Posterior Distribution
Given observed data $D$ with $k$ successes and $n-k$ failures:

- **Posterior Parameters**:
  $$\alpha_{\text{post}} = \alpha_{\text{prior}} + k$$
  $$\beta_{\text{post}} = \beta_{\text{prior}} + n - k$$

- **Posterior Distribution**:
  $$p(p \mid D) = \text{Beta}(p \mid \alpha_{\text{post}}, \beta_{\text{post}})$$

#### Interpretation
- **Updating Beliefs**: The posterior Beta distribution combines prior beliefs with observed data.
- **Flexibility**: By adjusting $\alpha$ and $\beta$, we can represent different levels of certainty.

![[Pasted image 20241123113706.png]]

### Making Predictions

#### Bayesian Predictive Distribution
To predict the probability of success in the next trial:
$$P(X_{\text{next}}=1 \mid D) = E[p \mid D] = \frac{\alpha_{\text{post}}}{\alpha_{\text{post}} + \beta_{\text{post}}}$$

With the Bayesian approach, average over all possible parameter settings. The prediction for heads happens to correspond to the mean of the posterior distribution. 

Given the posterior distribution, we can also answer other questions such as “what is the probability that π > 0.5 given the observed data?”.

## Model Comparison

### Comparing Models
Suppose we have two models for the coin:
- **Fair Coin Model**:
  - Assumes $p=0.5$.
  - No parameters to estimate.
- **Bent Coin Model**:
  - Assumes $p$ is unknown and uniformly distributed over $[0, 1]$.
  - Requires estimating $p$.

![[Pasted image 20241123120947.png]]

### Bayesian Model Comparison
- **Prior Probabilities**:
  $$P(\text{Fair}) = 0.8$$
  $$P(\text{Bent}) = 0.2$$

- **Compute Evidence**:
  - For the Fair Coin Model:
    $$P(D \mid \text{Fair}) = (0.5)^n$$
  - For the Bent Coin Model:
    $$P(D \mid \text{Bent}) = \int_0^1 P(D \mid p) P(p \mid \text{Bent}) dp$$

    Since $P(p \mid \text{Bent})$ is uniform, this simplifies to the Beta function.

- **Posterior Probabilities**:
  $$P(\text{Fair} \mid D) = \frac{P(D \mid \text{Fair}) P(\text{Fair})}{P(D)}$$
  $$P(\text{Bent} \mid D) = \frac{P(D \mid \text{Bent}) P(\text{Bent})}{P(D)}$$

  where $P(D)$ is the total probability of the data under both models.

### Decision Making
- **Model Selection**: Choose the model with the higher posterior probability.
- **Predictive Distribution**: Combine predictions from both models weighted by their posterior probabilities.


# 35-Discrete-Categorical-Distribution

We extend the concepts from binary variables to multiple discrete outcomes, which is essential in modeling categorical data such as word frequencies in text documents.

### Key Concepts
- **Discrete and Multinomial Distributions**: Modeling counts of multiple categories.
- **Dirichlet Distribution**: Prior distribution over multinomial parameters.

### Multinomial Distribution

#### Definition
The multinomial distribution generalizes the binomial distribution to more than two outcomes.

- **Parameters**:
  - $n$: Number of trials.
  - $p = [p_1, p_2, \dots, p_m]$: Probabilities of each category, where $\sum_{i=1}^m p_i = 1$.

#### Probability Mass Function
Given counts $k = [k_1, k_2, \dots, k_m]$ with $\sum_{i=1}^m k_i = n$:
$$P(k \mid n, p) = \frac{n!}{k_1! k_2! \dots k_m!} \prod_{i=1}^m p_i^{k_i}$$

#### Example: Rolling a Die
- **Outcomes**: $m=6$ faces.
- **Counts**: Number of times each face appears in $n$ rolls.

### Dirichlet Distribution

The Dirichlet distribution is a continuous multivariate probability distribution over the simplex of $m$-dimensional probability vectors $p$.

- **Parameters**: Concentration parameters $\alpha = [\alpha_1, \alpha_2, \dots, \alpha_m]$, with $\alpha_i > 0$.

#### Probability Density Function
$$\text{Dirichlet}(p \mid \alpha) = \frac{\Gamma\left(\sum_{i=1}^m \alpha_i\right)}{\prod_{i=1}^m \Gamma(\alpha_i)} \prod_{i=1}^m p_i^{\alpha_i - 1}$$

#### Properties
- **Conjugate Prior**: The Dirichlet distribution is the conjugate prior for the multinomial distribution.
- **Mean**:
  $$E[p_i] = \frac{\alpha_i}{\sum_{j=1}^m \alpha_j}$$

#### Posterior Distribution
Given observed counts $k$:
- **Posterior Parameters**:
  $$\alpha_{i, \text{post}} = \alpha_{i, \text{prior}} + k_i$$
- **Posterior Distribution**:
  $$p(p \mid k) = \text{Dirichlet}(p \mid \alpha_{\text{post}})$$

#### Symmetric Dirichlet Distribution
- **Definition**: A Dirichlet distribution where all concentration parameters are equal: $\alpha_i = \alpha$ for all $i$.
- **Interpretation**:
  - Small $\alpha$: Distributions are more variable; samples are likely to be sparse.
  - Large $\alpha$: Distributions are more uniform; samples are more balanced.

### Sampling from the Dirichlet Distribution
- **Method**:
  - Sample $m$ independent gamma random variables $g_i$ with shape parameter $\alpha$ and scale parameter 1.
  - Normalize:
    $$p_i = \frac{g_i}{\sum_{j=1}^m g_j}$$

## Application: Word Counts in Text

### Modeling Word Frequencies
- **Vocabulary**: Set of $m$ distinct words.
- **Document Representation**: Counts $k = [k_1, k_2, \dots, k_m]$ of each word in a document.
- **Multinomial Distribution**: Models the probability of observing these counts given word probabilities $p$.

### Prior Over Word Probabilities
- **Dirichlet Prior**: Encodes prior beliefs about word probabilities.
- **Posterior Updating**: Update the Dirichlet parameters based on observed word counts.

### Implications for Text Modeling
- **Flexibility**: Can model documents with varying word distributions.
- **Bayesian Inference**: Allows incorporation of prior knowledge and uncertainty.
