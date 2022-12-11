# Evaluating Inverse Contextual Bandits against model-based inference

This repository contains the necessary code to replicate the results presented in my report *'Evaluating Inverse Contextual Bandits against model-based inference'* based on an ICML 2022 paper *'Inverse Contextual Bandits: Learning How Behavior Evolves over Time'*.

The original work of Alihan Hüyük et al. is available at this [github repository](https://github.com/alihanhyk/invconban).

My proposed modifiaction of the *Bayesian ICB* (B-ICB) for the optimistic and greedy policies are available in `src/main-optimistic-bicb.py` and `src/main-greedy-bicb.py`.

Main experimental results can be obtained by running 

```
./run.sh
python src/main-mle-model-and-eval.py 
python src/optim-greedy-bicb.py 
```

Inference algorithms for the stationary, linear, stepping and regressing models together with their evaluation (part of Tables 1 and 2) are available in `src/main-mle-model-and-eval.py` 

Ealuation of the original, optimistic and greedy versions of B-ICB (second part of Table 1 and 2, Figure 1, and Table 3) can be found in `src/optim-greedy-bicb.py`.

**Note**: In order to run the experiments access to semi-synthetic data as generated in the original paper is needed.

