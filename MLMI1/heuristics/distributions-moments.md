10 October 

### Moments of a Distribution

1. **First Moment (Mean):**
   - The **first moment** of a distribution is its **mean**. It represents the **central location** or the **average value** of the data. For a random variable $X$, the mean is given by: $$ \mu = \mathbb{E}[X] = \int_{-\infty}^{\infty} x f(x) dx $$ where $f(x)$ is the probability density function (PDF) of $X$.


2. **Second Moment (Variance):**
   - The **second moment** is related to the **variance**, which measures the **spread** or **dispersion** of the distribution around the mean. The variance is defined as:$$ \text{Var}(X) = \mathbb{E}[(X - \mu)^2] = \int_{-\infty}^{\infty} (x - \mu)^2 f(x) dx $$
3. **Skewness (Third Moment):**
   - **Skewness** measures the **asymmetry** of the distribution. A symmetric distribution like the normal distribution has a skewness of 0, while distributions that are skewed have non-zero skewness. Skewness is given by the third standardized moment:
     $$ \text{Skewness}(X) = \frac{\mathbb{E}[(X - \mu)^3]}{\sigma^3} $$
   - **Positive skewness** indicates a distribution with a longer right tail (data are more spread out to the right of the mean).
   - **Negative skewness** indicates a distribution with a longer left tail (data are more spread out to the left of the mean).

4. **Kurtosis (Fourth Moment):**
   - **Kurtosis** measures the **tailedness** or **sharpness of the peak** of the distribution. It is defined by the fourth standardized moment:
     $$ \text{Kurtosis}(X) = \frac{\mathbb{E}[(X - \mu)^4]}{\sigma^4} $$
   - A normal distribution has a kurtosis of 3, which is considered **mesokurtic**.
   - **Excess kurtosis** is often used, where we subtract 3 from the kurtosis value:
     $$ \text{Excess Kurtosis} = \text{Kurtosis} - 3 $$
   - **Leptokurtic** distributions (positive excess kurtosis) have heavy tails and a sharp peak, while **platykurtic** distributions (negative excess kurtosis) have lighter tails and a flatter peak.


### Interpretation of the Student-t Distribution as an Infinite Mixture Model

The **Student-t distribution** can be interpreted as an **infinite mixture of normal distributions** with different variances. This means that a t-distribution is not a fixed distribution but rather a **weighted combination of normal distributions** with varying levels of uncertainty.

#### 1. **Mixture of Normals with Random Variance:**
   - The t-distribution arises in Bayesian statistics when you assume that the data follow a normal distribution, but there is **uncertainty about the variance**. Specifically, if you assume that the variance follows an **inverse gamma distribution**, the marginal distribution of the data (after integrating over all possible variances) is a Student-t distribution.
   - Formally, if you assume that the data $X$ follows a normal distribution, $X \sim \mathcal{N}(\mu, \sigma^2)$, but the variance $\sigma^2$ is itself drawn from an inverse gamma distribution, then the marginal distribution of $X$ is:
     $$ X \sim \text{Student-t}(\nu) $$

#### 2. **Formula for the Student-t Distribution:**
   The probability density function (PDF) of the Student-t distribution with $\nu$ degrees of freedom is given by:
   $$ 
   f(t|\nu) = \frac{\Gamma\left(\frac{\nu + 1}{2}\right)}{\sqrt{\nu \pi} \Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{t^2}{\nu}\right)^{-\frac{\nu + 1}{2}} 
   $$
   where:
   - $\Gamma(\cdot)$ is the Gamma function,
   - $\nu$ is the degrees of freedom,
   - $t$ is the variable.

### Laplace Distribution

The **Laplace distribution** (also known as the **double exponential distribution**) is a continuous probability distribution characterized by a peak at its mean and symmetric exponential tails. It is useful for modeling data with heavier tails than the normal distribution, which makes it robust to outliers.

#### 1. **Probability Density Function (PDF):**
   The probability density function (PDF) of the Laplace distribution is given by:
   $$ 
   f(x|\mu, b) = \frac{1}{2b} \exp\left(-\frac{|x - \mu|}{b}\right) 
   $$
   where:
   - $\mu$ is the **location parameter** (mean), indicating the peak of the distribution,
   - $b$ is the **scale parameter**, controlling the spread of the distribution,
   - $x$ is the variable.
   
   The Laplace distribution has a peak at $x = \mu$, and the probability decreases exponentially as you move away from the mean.


### Heteroscedasticity vs. Homoscedasticity

#### 1. **Heteroscedasticity:**
   - **Definition**: Heteroscedasticity occurs when the **variance** of the errors (or residuals) in a model **varies** across different levels of the independent variable(s). In other words, the spread of the data is not constant as the value of the predictor changes.
   - **Key Feature**: The error terms have **non-constant variance**, meaning that some parts of the data exhibit more variability than others.
#### 2. **Homoscedasticity:**
   - **Definition**: Homoscedasticity is when the **variance** of the errors (or residuals) in a model remains **constant** across all levels of the independent variable(s). In this case, the spread of the data points is the same regardless of the predictor's value.
   - **Key Feature**: The error terms have **constant variance**.
