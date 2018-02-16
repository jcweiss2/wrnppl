### Wavelet reconstruction networks

This a repository for the working paper: "Machine Learning for Clinical Risk: Wavelet Reconstruction Networks for Marked Point Processes". The paper is available at:
[wrnppl](https://www.andrew.cmu.edu/user/jweiss2/180208clinicalrisk2.pdf). Please cite with the [bib](https://www.andrew.cmu.edu/user/jweiss2/resources/wrnppl.bib).

Wavelet reconstruction networks are a neural network that generalizes Hawkes processes. In essence, it combines the relative timing and values of features with reduction functions to estimate a step-wise approximation to the hazard (the rate) of an outcome of interest. It captures the timing with 1-d wavelet reconstructions and timing-and-value with 2-d reconstructions.

Input: a four-column CSV (possibly train/tune/test), and a bunch of arguments, e.g., indicating the target event and settings of the model.
Output: rate predictions for the target event and a neural network model