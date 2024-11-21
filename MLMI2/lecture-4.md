# Acoustic Modelling for Large Vocabulary Speech Recognition

## Introduction
Automatic Speech Recognition (ASR) systems aim to transcribe spoken language into text. One of the critical components of an ASR system is the acoustic model, which represents the relationship between audio signals and the linguistic units (such as phonemes or words) they represent. In large vocabulary speech recognition tasks, the challenge lies in accurately modeling a vast number of possible words and their variations due to context, speaker differences, and environmental factors.

This masterclass delves into the strategies for effective acoustic modeling in Hidden Markov Model (HMM)-based large vocabulary speech recognition systems. We will explore the selection of modeling units, the handling of context dependency, model structures, training procedures, and advanced techniques such as discriminative sequence training.

---

## 1. Acoustic Modelling Units

### 1.1. Criteria for Selecting Units
When choosing the basic units to model speech in an ASR system, several criteria must be considered:

- **Compactness:** The units should be manageable in number, even for large vocabularies, to ensure computational efficiency.
- **Simple Mapping:** There should be a straightforward relationship between recognition units and words or sentences to facilitate decoding.
- **Variability Handling:** Units should account for speech variability, including linguistic structures and co-articulation between neighboring sounds.
- **Extendability:** The system should be able to handle words not seen in the training data (out-of-vocabulary words).
- **Well-Defined Units:** Units must be easily identifiable and consistently defined to allow for large-scale data collection and training.

An additional consideration is the **trainability** of the units. The chosen units should enable the creation of accurate models without overfitting, balancing the number of parameters with the available training data.

---

### 1.2. Words as Units
Using whole words as the basic units in acoustic modeling is impractical for large vocabulary tasks due to several reasons:

- **Scalability Issues:** The number of words in a language can be enormous (e.g., over 65,000 in a typical vocabulary), leading to an unmanageable number of models.
- **Data Requirements:** Each word model would require sufficient training data to estimate its parameters accurately, which is often infeasible.
- **Out-of-Vocabulary Words:** Words not present in the training data cannot be modeled, limiting the system's ability to recognize new words.
- **Co-Articulation Effects:** While intra-word variations can be captured, inter-word co-articulation (the influence of surrounding words on pronunciation) may not be adequately modeled.

Due to these limitations, **sub-word units** are typically preferred in acoustic modeling for large vocabulary ASR systems.

---

## 2. Possible Speech Units

### 2.1. Phones
Phones are the distinct speech sounds in a language. In English, there are approximately 40 to 50 phones, making them a compact set for modeling. Advantages of using phones include:

- **Compactness:** A manageable number of units simplifies the model.
- **Well-Defined:** Phones are linguistically defined and can be consistently identified.
- **Extendability:** By combining phones, any word (including unseen words) can be modeled.

However, phones are highly context-dependent due to co-articulation and other phonetic variations, which can affect their acoustic realization.

---

### 2.2. Syllables
Syllables represent larger units of speech that consist of one or more phones. English has over 10,000 syllables, which poses challenges:

- **Large Number:** The sheer number of syllables makes modeling and parameter estimation difficult.
- **Data Requirements:** Many syllables may have insufficient training data for accurate modeling.
- **Variability:** Syllable pronunciation can vary significantly based on context and speaker.

While syllable-based models can capture certain prosodic features and co-articulation effects, they are less commonly used in English ASR systems compared to phone-based models. In languages with a simpler syllabic structure (e.g., Mandarin Chinese), syllable-based units may be more practical.

---

### 2.3. Characters and Word-Piece Units
- **Characters (Graphemes):** Represent the written symbols of a language. Using characters as modeling units can simplify the mapping between acoustic models and text, eliminating the need for a pronunciation dictionary. However, in languages like English, the relationship between characters and pronunciation is complex and inconsistent (e.g., silent letters, multiple pronunciations).

- **Word-Piece Units:** Sub-word units derived from statistical analysis of text corpora, such as byte pair encoding (BPE). They can capture common morphemes, prefixes, suffixes, and frequent word fragments, balancing the granularity between characters and words.

Word-piece models are increasingly popular in end-to-end ASR systems due to their ability to handle out-of-vocabulary words and model longer dependencies.

---

## 3. Model Structures and Pronunciations

### 3.1. Hidden Markov Model (HMM) Structures
In HMM-based acoustic modeling, each basic unit (e.g., a phone) is represented by an HMM. A standard structure for phone HMMs includes:

- **Three Emitting States:** Each state models a portion of the phone's duration (beginning, middle, end).
- **Left-to-Right Architecture:** States are arranged sequentially with no skips, reflecting the temporal progression of speech sounds.

This simple yet effective structure captures the temporal dynamics of speech production for each phone.

---

### 3.2. Pronunciation Dictionaries
A pronunciation dictionary maps words to sequences of phones (phonetic transcriptions). For example, the word "the" can have multiple pronunciations:

- $\text{/dh ax/}$
- $\text{/dh iy/}$

Multiple pronunciations account for variations due to dialects, speaking styles, or context. During training, models are typically trained using one selected pronunciation per word, often the most common or contextually appropriate one. Alternatively, a network of possible pronunciations can be used to represent variability.

---

### 3.3. Handling Pronunciation Variability
Pronunciation variability poses challenges in acoustic modeling:

- **Overfitting Risks:** Maximum Likelihood Estimation (MLE) can lead to overfitting, especially when limited data is used to estimate a large number of parameters.
- **Variance Floor:** To prevent variances from becoming too small (leading to overconfidence), a variance floor can be set for Gaussian components in a Gaussian Mixture Model (GMM).
- **Bayesian Estimation:** Incorporating prior distributions can regularize parameter estimates but requires appropriate priors.
- **Variable Number of Components:** Adjusting the number of mixture components based on state occupancy can help balance model complexity with data availability.

---

## 4. Phone Variation and Context Dependency

### 4.1. Co-Articulation Effects
Co-articulation refers to the influence of surrounding sounds on the pronunciation of a phone. The acoustic realization of a phone can vary significantly depending on its phonetic context.

For example, in the phrase "We were away with William in Sea World," the realization of the phone $/w/$ differs across occurrences due to varying contexts. The two $/w/$ sounds in the same triphone context (e.g., $/w/$ preceded by $/e/$ and followed by $/i/$) are more acoustically similar than those in different contexts.

### 4.2. Context-Dependent Phone Models
To account for variability due to co-articulation, context-dependent phone models are used. These models consider the phonetic context of each phone, leading to more accurate acoustic representations.

#### 4.2.1. Biphones
- **Left Biphones:** Consider the left context (preceding phone).  
  Example: $/s-p/$ (phone $/p/$ preceded by $/s/$)
- **Right Biphones:** Consider the right context (following phone).  
  Example: $/p+iy/$ (phone $/p/$ followed by $/iy/$)

#### 4.2.2. Triphones
- Consider both left and right contexts.  
  Example: $/s-p+iy/$ (phone $/p/$ preceded by $/s/$ and followed by $/iy/$)

Using triphones allows for more precise modeling but dramatically increases the number of models.

---

### 4.3. Challenges with Context-Dependent Models
The main challenge with context-dependent models is **trainability**. The number of possible contexts grows exponentially with the context length:

- **Monophones:** $N$ models (e.g., 40-50 phones)
- **Biphones:** $N^2$ models
- **Triphones:** $N^3$ models

For English with $N=50$, there could be up to 125,000 triphones. However, not all possible triphones occur in the language or are present in the training data.

---

## 5. System Trainability and Parameter Estimation

### 5.1. Balancing Detail and Robustness
A critical aspect of acoustic modeling is balancing the level of acoustic detail (model specificity) with robust parameter estimation (model generalization). Overly specific models may not have enough training data, leading to poor generalization on unseen data.
### 5.2. Backing Off Strategies
Backing off involves reverting to less specific models when there is insufficient data to train highly specific ones:

$$\text{Triphone} \rightarrow \text{Biphone} \rightarrow \text{Monophone}$$

This approach can lead to inconsistencies and abrupt changes in model specificity, affecting recognition performance.

### 5.3. Parameter Sharing (Tying)
Parameter sharing allows models to share parameters across acoustically similar contexts, improving robustness:

- **Flexibility:** Models of the same complexity can share data, enhancing parameter estimation.
- **Levels of Sharing:**
  - **State Level:** Sharing parameters at the state level across different models.
  - **Mixture Components:** Sharing Gaussian components in GMMs across models (tied mixtures).

Parameter sharing reduces the number of unique parameters, alleviating the data scarcity problem for rare contexts.

### 5.4. Smoothing and MAP Estimation
Smoothing combines estimates from specific models with those from more general ones to improve robustness. Maximum A-Posteriori (MAP) estimation incorporates prior knowledge (e.g., from context-independent models) into parameter estimation, providing a principled way to balance specificity and robustness.

---

## 6. Constructing Context-Dependent Models

### 6.1. Bottom-Up Clustering
Bottom-up clustering starts with highly specific models and merges them based on similarity:

1. **Initialize Models:** Train models for all observed contexts.
2. **Compute Similarities:** Measure acoustic similarity between models (e.g., using Kullback-Leibler divergence).
3. **Merge Models:** Combine similar models to form clusters.
4. **Iterate:** Repeat merging until a stopping criterion is met (e.g., minimum data threshold).

#### Limitations:
- **Unreliable for Rare Contexts:** Models for infrequent contexts may not merge appropriately.
- **Data Dependency:** Cannot handle unseen contexts not present in the training data.

---

### 6.2. Top-Down Decision Tree Clustering
Top-down decision tree clustering provides a systematic way to cluster contexts using phonetic questions:

1. **Initialize:** Start with a single cluster containing all contexts.
2. **Define Questions:** Create a set of yes/no questions about phonetic properties (e.g., "Is the left phone a nasal?").
3. **Split Clusters:** At each node, select the question that best increases the likelihood of the training data when splitting the cluster.
4. **Grow the Tree:** Recursively apply splitting to child nodes until stopping criteria are met (e.g., minimum data threshold per leaf).

#### Advantages:
- **Handles Unseen Contexts:** Assigns unseen contexts to existing clusters based on phonetic properties.
- **Incorporates Expert Knowledge:** Phonetic questions can encode linguistic insights.
- **Flexible Granularity:** Allows for varying levels of specificity based on data availability.

#### Disadvantages:
- **Greedy Algorithm:** Decisions are locally optimal and may not yield the global optimum.
- **Computational Complexity:** Requires efficient algorithms to handle large datasets and model complexities.

---

## 7. Decision Tree-Based State Tying

### 7.1. Clustering at the State Level
Instead of clustering entire models, state-level clustering focuses on individual HMM states. This allows for more fine-grained parameter sharing:

- **Separate Trees for Each State Position:** Build a decision tree for each state position (e.g., beginning, middle, end) of each phone.
- **Context Questions:** Use phonetic questions relevant to the specific state.
- **State Clusters:** Resulting leaf nodes represent tied states sharing parameters.

---

### 7.2. Constructing the Decision Tree
The tree is built by maximizing the increase in the approximate likelihood of the training data when splitting clusters. At each node:

1. **Calculate Statistics:** Gather sufficient statistics (e.g., means, variances, occupation counts) for the current cluster.
2. **Evaluate Questions:** For each potential question, compute the change in log-likelihood if the cluster is split based on that question.
3. **Select the Best Split:** Choose the question that yields the maximum increase in likelihood.

#### Stopping Criteria:
- The increase in likelihood falls below a threshold.
- The resulting clusters have insufficient data (below a minimum occupation count).

---

### 7.3. Advantages of State Tying
- **Efficient Use of Data:** Allows rare contexts to benefit from shared parameters.
- **Model Compactness:** Reduces the total number of unique parameters.
- **Improved Generalization:** Enhances robustness to unseen data.

---

## 8. Training Procedures for Continuous Speech Recognition

### 8.1. Isolated Word Training
Initially, HMMs can be trained using isolated word utterances. However, this approach does not capture the variability present in continuous speech, such as co-articulation between words.

---

### 8.2. Viterbi Alignment and Segmentation
Viterbi alignment involves:

1. **Building Composite Models:** Concatenate HMMs corresponding to the sequence of words in an utterance.
2. **Alignment:** Use the Viterbi algorithm to find the most likely state sequence, effectively segmenting the utterance.
3. **Parameter Estimation:** Train models based on the segmented data.

This process can be iterative, refining the models and segmentations in each iteration.

---

### 8.3. Sentence-Level Baum-Welch Training (Embedded Training)
Sentence-level Baum-Welch training incorporates the Baum-Welch algorithm (the Expectation-Maximization algorithm for HMMs) at the utterance level:

- **Composite HMMs:** Construct HMMs for entire utterances based on the known word sequences.
- **Forward-Backward Algorithm:** Compute the expected state occupancies and transitions without committing to a single segmentation.
- **Parameter Re-estimation:** Update model parameters based on accumulated statistics over all training data.

#### Advantages:
- **Soft Alignment:** Accounts for uncertainty in segmentation, leading to more robust models.
- **Efficiency:** Processes entire utterances, capturing continuous speech characteristics.

---

### 8.4. Flat Start Initialization
Flat start initializes models with uniform parameters (e.g., equal probabilities, identical Gaussians). Training then proceeds using embedded training without relying on pre-segmented data or pre-trained models.

---

## 9. Building a GMM-HMM Large Vocabulary System

### 9.1. Step-by-Step Procedure

1. **Initialization:**
   - Use existing models (e.g., from TIMIT dataset) to obtain initial phone alignments.
2. **Monophone Training:**
   - Train context-independent (CI) monophone models with a single Gaussian component per state.
3. **Context Expansion:**
   - Clone monophones to create context-dependent (CD) models for all observed cross-word triphones.
4. **Unclustered Triphone Training:**
   - Train single Gaussian CD triphone models without parameter sharing.
5. **Decision Tree Clustering:**
   - Perform state-level decision tree clustering to tie states and share parameters.
6. **Re-estimation:**
   - Retrain models using the new state clusters.
7. **Model Complexity Increase:**
   - Incrementally increase the number of Gaussian mixture components per state (e.g., from 1 to 12) using a "mixing-up" procedure.
   - Retrain models at each level of complexity.

---

### 9.2. Example System Details
- **Training Data:** Wall Street Journal corpus (approx. 66 hours, 36,000 sentences).
- **Features:** 12 Mel-Frequency Cepstral Coefficients (MFCCs), log-energy, delta, and delta-delta features.
- **Models:** Gender-independent and gender-dependent GMM-HMMs with 12 Gaussian components per state.
- **Vocabulary:** 65,000 words with multiple pronunciations.
- **Language Model:** Trigram model.

#### Performance:
Word Error Rates (WER) of approximately 9-10% on development and evaluation sets.

---

## 10. Introduction to Discriminative Sequence Training

### 10.1. Limitations of Maximum Likelihood Estimation (MLE)
While MLE aims to maximize the likelihood of the observed data given the model, it assumes:

1. **Infinite Training Data:** In practice, training data is limited.
2. **Model Correctness:** The model may not perfectly represent the true data-generating process.

These assumptions often do not hold in real-world ASR systems, leading to suboptimal performance.

---

### 10.2. Maximum Mutual Information Estimation (MMIE)
MMIE focuses on maximizing the posterior probability of the correct transcription given the observed data:

$$F_{\text{MMIE}}(\lambda) = \sum_{r=1}^R \log \frac{p_\lambda(O_r \mid M_{H_r}) P(H_r)}{\sum_H p_\lambda(O_r \mid M_H) P(H)}$$

- **Numerator:** Likelihood of the data given the correct transcription.
- **Denominator:** Summation over all possible transcriptions (competing hypotheses).
- **Objective:** Increase the probability of the correct transcription while decreasing the probability of incorrect ones.

---

### 10.3. Advantages of MMIE
- **Discriminative Training:** Directly optimizes for better discrimination between correct and incorrect hypotheses.
- **Closer to Recognition Performance:** More closely related to minimizing word error rates.

---

### 10.4. Optimization Challenges
- **Computational Complexity:** Computing the denominator requires evaluating all possible transcriptions.
- **Lattice Approximations:** Use word lattices to approximate the denominator efficiently.
- **Optimization Methods:** Extended Baum-Welch algorithms for GMM-HMMs and gradient-based methods for DNN-HMMs.

---

### 10.5. Error-Based Objective Functions
**Minimum Phone Error (MPE)** and **Minimum Word Error (MWE)** training aim to directly minimize expected error rates:

- **MPE:** Weights hypotheses based on their phone-level errors.
- **MWE:** Focuses on word-level errors, more closely aligning with the ASR evaluation metric.

These methods further enhance recognition performance by focusing on the ultimate goal of reducing errors.

---

## 11. Character-Based and Word-Piece Units

### 11.1. Character (Grapheme) Models
#### Advantages:
- **No Pronunciation Dictionary:** Simplifies the modeling pipeline.
- **Extendability:** Can model any word, including out-of-vocabulary words.

#### Disadvantages:
- **Pronunciation Variability:** English has a complex relationship between spelling and pronunciation.
- **Context Dependency:** Characters may require extensive context modeling to capture pronunciation variations.

---

### 11.2. Context-Dependent Grapheme Models
To address variability, context-dependent grapheme models can be employed:

- **Longer Contexts:** Incorporate longer-range dependencies.
- **Decision Trees:** Use phonetic questions adapted for graphemes.

---

### 11.3. Word-Piece Models
#### Construction:
- **Statistical Methods:** Derive units based on frequency and co-occurrence statistics (e.g., Byte Pair Encoding).
- **Controlled Vocabulary Size:** Balance between granularity and model complexity.

#### Advantages:
- **Flexibility:** Can represent common morphemes and word fragments.
- **Full Coverage:** Able to model any word in the language.

#### Applications:
- **End-to-End ASR Systems:** Used in models that jointly learn acoustic and language models without intermediate phonetic representations.
- **Natural Language Processing:** Common in language models like BERT and GPT.

---

## 12. Summary
In large vocabulary speech recognition, effective acoustic modeling is essential for accurate transcription. Key strategies include:

1. **Selecting Appropriate Units:**
   - Phones are commonly used due to their balance of compactness and extendability.
   - Context-dependent models capture variability due to co-articulation.
2. **Managing Model Complexity:**
   - Parameter sharing through state tying and decision tree clustering enables robust estimation despite data limitations.
3. **Advanced Training Techniques:**
   - Discriminative sequence training methods like MMIE and MPE improve performance by focusing on recognition accuracy rather than just data likelihood.
4. **Exploring Alternative Units:**
   - Characters and word-piece models offer advantages in end-to-end systems, simplifying the modeling process and accommodating unseen words.

Understanding and implementing these concepts allows for the development of sophisticated ASR systems capable of handling the complexities of continuous, natural speech in large vocabularies.

---

## References
1. **HTK Book:** Comprehensive guide on Hidden Markov Model Toolkit (HTK) used for building ASR systems.
2. **Jurafsky, D., & Martin, J. H. (2009):** *Speech and Language Processing.* Pearson Prentice Hall.
3. **Povey, D., & Woodland, P. C. (2002):** Minimum Phone Error and I-smoothing for Improved Discriminative Training. *Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP).*
4. **Sennrich, R., Haddow, B., & Birch, A. (2016):** Neural Machine Translation of Rare Words with Subword Units. *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL).*

