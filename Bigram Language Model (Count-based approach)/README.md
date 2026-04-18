## Overview

This is a sequel to one of my side mini projects. It's a bigram language model that predicts the next character based on the previous one. This project implements a character-level bigram language model from scratch.

A bigram model estimates:

> the probability of the next character given the current character

Formally:

> P(x<sub>t</sub>​∣x<sub>t−1</sub>​)

The model is built using simple frequency counting and normalization, without any machine learning libraries. 

## Key highlights

* **Count-based probability modeling** Constructed a bigram model by computing frequency counts of character transitions and converting them into conditional probabilities.
    
* **Matrix-based representation**
Represented the model as a 2D transition matrix where each row encodes a probability distribution over the next character.

* **Negative Log-Likelihood (NLL) evaluation** 
Implemented a loss function to quantitatively measure how well the model predicts observed data.

* **Add-one smoothing**
Applied smoothing to handle zero-probability issues.

* **Data visualization**
Used heatmaps to visualize character transition patterns and better understand learned distributions.
* **Understanding model limitations**
Observed that bigram models capture only local dependencies and fail to model long-range structure in language.