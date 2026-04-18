## Overview

This is a sequel to one of my side mini projects. It's a bigram language model that predicts the next character based on the previous one. This project implements a character-level bigram language model from scratch.

A bigram model estimates:

> the probability of the next character given the current character

Formally:

> P(x<sub>t</sub>​∣x<sub>t−1</sub>​)

The model is built using simple frequency counting and normalization, without any machine learning libraries. 