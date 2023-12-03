# @November 30, 2023

Created: November 30, 2023 2:47 PM

Outline for my talk on December 7, 2023

Each horizontal rules is a slide

### Title

Analysis of Emergent Convolutional Structure in a Two-Layer Neural Network

## Introduction

---

### Structure in NNs

- Neural networks trained with SGD often learn very stable and structured internal representations
    - Canonical example is the emergence of Gabors in first layer of CNNs…
- This is even true for less architecturally constrained models like FCNs
- But most analytical works aren’t really concerned with understanding these representations, they focus more on the loss
    - Like the neural tangent kernel, where there is no feature learning
    - Or, certain mean-field analyses (cite Mei?)

---

### Structure in NNs

- There are exceptions
    - Student-teacher (explain what this is), but this assumes specific (Gaussian) input, and a lot of interesting internal structure that emerges in NNs is driven by the non-Gaussianity of inputs (e.g. naturalistic images are very non-Gaussian, they contain edges and are translation- and scale-invariant)
    - Another approach is to consider linear activation, which can capture a lot of interesting behavior seen in neural networks, but cannot solve problems beyond linear classification
    - Gated linear networks address this to some degree (e.g. they can solve XOR), but, to put it succinctly, they assume the way the nonlinearity interacts with the data doesn’t change during training, which is not true in general and cannot capture emergent structures like the ones I mentioned above
- In summary, there isn’t much you can say analytically about structure that emerges in neural nets as a result of non-Gaussian inputs
    - This is despite such emergent structures being very stable and existing even in very simple settings

---

### A minimal example

- One of the simplest settings we’ve found is from Alessandro Ingrosso & Sebastian Goldt (2022?), who constructed a special data model where they could tune just a single parameter that controls the degree of non-Gaussianity, and they showed that this parameter could control whether a FCN trained with SGD learned a convolutional structure
    - this was invariant to activation function and how many learnable layers the network had
    - whether inputs were 1-D or 2-D
- Their data model:
    
    $$
    \operatorname{erf}
    $$
    
    - translation-invariant
    - symmetric about 0 (makes it impossible to solve linearly)
    - show examples from this distribution

---

### A minimal example

- Introduce the neural network they mostly focus on:
    - two-layer, sigmoid activation, learnable bias
    - then restrict second-layer to be $\frac{1}{K}$
- Show the resulting RFs, in the imshow() view but also single & grouped RFs using plot()
- Describe the three structural phenomena: frequency specialization, tiling, and localization

---

### Alessandro’s explanation

- The message of Alessandro’s paper is that non-Gaussian data statistics are responsible for this interesting behavior
- They aren’t able to make much analytical progress, but they do try by considering a one-neuron model with sigmoid activation
    - (show this math)
- they also use GET to show that it’s non-Gaussianity in the pre (or post?) activations that’s associated with the emergence of the interesting behavior

---

### Alessandro’s explanation

- They also used a third-order Taylor approximation to show second-order moments are not sufficient for localization
- They decompose the fourth-order moment tensor and showed that its principal components look localized as $g$ gets large
- They noted that the size of localized region in the components was similar to $\xi$
- This was all observational, they couldn’t directly connect any of these parameters to localization
    - The takeaway was that it’s non-Gaussianity that drives the convolutional structure that is learned, but how it does so remained elusive
- That’s what we tried to understand
    - We wanted to understand how these three structural properties—frequency specialization, tiling, and localization—are driven by non-Gaussianity features, and what exactly “non-Gaussianity” means

---

## Results

---

### Frequency specialization

- Emerges early in training
- Easy to characterize; well, it’s easy to predict, at least

---

### Tiling - what to say here??

- Also emerges very early on, but hard to characterize analytically
- Simulations show that this emerges in Gaussian case
- Simulations also show that where it tiles in Gaussian case is similar to where it tiles in non-Gaussian case

---

### Localization

- Once weight-sharing emerges, we can analyze one receptive field at a time
- Universal perspective
    - What is $f(w)$ doing?
    - What is the $(\Sigma_0 + \Sigma_1) w$ term doing?
        - Weighting $\Sigma_1$ more yields more *Mexican-hat-like* receptive field
        - Weighting $\Sigma_0$ more yields more *spread out* receptive field
- Marginals need to have density not near zero
    - Show equation
- NLGP as $g \to \infty$ w/ Gaussian kernel
- Single pulse (marginals on $\pm 1$)
- Ising model
- Single pulse (marginals that vary)

---

## Ongoing Work

---

- Trying to understand which assumptions on the data distribution can be relaxed to still yield similar analytical results
    - Specifically, can $\tilde{\mu}_{\mid X_i}$ and $\tilde{\Sigma}_{\mid X_i}$ always be written in terms of $\Sigma_1$ and $k_1$? If not, what can we say about them?
    - When does $f(w)$ not lose the localization term inside $\Phi(\cdot)$?
        - *Why does it disappear in the Gaussian case?*
        - *Why doesn’t it disappear in the high-gain case?* An incomplete answer to this is that it’s because the high-gain data has more mass away from zero. But so could an elliptical distribution. So what gives? What about elliptical distributions leads the term to disappear?
    - When does the Gaussian approximation hold? Does it only work when $k_1$ has a certain bandwidth?
    - Can we get rid of symmetry about zero? How about translation-invariance?

### Future Work

- Consider more general data invariances
    - How is translation-invariance important for:
        1. *symmetry breaking*: 
        2. *tiling*: 
        3. *localization*: translation-invariance gives us convolutional smoothing, but is this essential? Could we do it with something else?
- Why does high-gain trained model always do better?
- Negative results?
    - GDLN?