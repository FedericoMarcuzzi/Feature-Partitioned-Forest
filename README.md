# Feature-Partitioned-Forest

Creatd by Federico Marcuzzi

Università Ca' Foscari Venezia


The code in this repository was used to collect the results of the experiments defined in Robust Tree Ensembole against Adversarial Examples master’s degree.


abstract:

Machine learning models are subject to attacks that undermine security and can generate unexpected behaviors.
To the detriment of other machine learning algorithms, little has been done so far to create models based on decision trees that are robust to these attacks.
The present thesis proposes a new method to train a model based on an ensemble of decision trees, robust to specific attacks.
In particular, we developed a precise way of splitting the features of the dataset between the base-learners within the ensemble.
This split guarantees a robustness by construction against evasion attacks.
The threat model we defined limits the attacker’s strength to a maximum number of modifiable features.
The constraint forces the attacker to generate adversarial examples that differ from the original instance at most b features.
Also together with the model we proposed two approximation algorithms to certify a lower bound robustness of the model.
These approximation algorithms have significantly less computational complexity than testing the robustness of the model with a brute-force attack strategy.
We have compared the robustness of our algorithm with Random Forest and with some robust tree ensemble methods taken from the state of the art of adversarial example.
The experiments led to two important results.
First, the comparison showed that the robustness of our model is higher than that of the other models tested.
Secondly, the two robustness certification algorithms do not differ much from the real robustness of the model.
