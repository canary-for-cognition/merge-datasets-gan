# merge-datasets-gan
A Generative Adversarial Network (GAN) for learning a shared representation for different datasets

## Background

- Alzheimer’s Disease (AD) is one of the most widespread and costliest illnesses in first world countries
- No treatment available to permanently cure AD
  - However, effects of AD progresses in stages
  - Gravity of symptoms increases incrementally as time passes
- It’s crucial to identify AD as soon as possible to mitigate symptoms and test new treatments (which mostly apply to the first stages only)
- Screening tools based on Machine Learning (ML) show promising accuracy and increased efficiency in terms of time and human-resources

## Motivation

- Multiple different datasets for AD classification exist (e.g., CANARY, [DementiaBank](https://dementia.talkbank.org/), etc.)
  - Them all have differences (e.g., data collection strategy, modality, etc.)
- All are relatively small for training ML and especially DL models effectively
- There is a lack of large scale datasets related to AD hindering progress of research on automatic AD screening

## Proposed Approach and Method
- Try to combine data stemming from different sources (datasets) building a shared representation that can be fed to a classifier
  - Very ambitious goal (high risk but high reward)
- How do we build the shared representation? Using a GAN!
  - Train encoder to learn representations of data from different datasets
  - Train discriminator to recognize from which dataset the representation was originally derived
  - When the discriminator cannot make proper distinction anymore then it means representations are sufficiently similar (i.e., homogeneous) and can be used all together as data points for classifier
    - Note: do not want all our data points to be equal, crucial to learn homogeneous representation preserving individual info from each datapoint (big challenge!)

## GAN Architecture for Shared Representation Learning

![image](https://user-images.githubusercontent.com/23050671/144095179-09b89c0f-f957-4cb5-8ca0-3a8a309210d5.png)
