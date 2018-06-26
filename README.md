# neural-coding-of-probabilities
This contains the code related to my master thesis project about the characterization of neural coding of probabilities.

Three different simulations can be run:
1) Simulation 1: simplification of encoding models, r² matrix on N_fit and N_true
2) Simulation 2: identifiability confusion matrix
3) Simulation 3: difference between identifiability confusion matrices from transition and Bernoulli probabilities. 


The Python scripts are:

- simulation1.ipynb: Jupyter notebook with all the processes of simulation 1
- feature _creation1.py: script to make the design matrix X for simulation 1  
- cross_validation1.py: script running the cross-validation for simulation 1 (for use on the cluster)

- simulation2.ipynb: Jupyter notebook with all the processes of simulation 2 and 3

- plot_activities.py: script giving neural activity plots for each coding schemes.
- neural_proba.py: contains main classes and functions useful for sequential data importation, activity/BOLD conversion. 
- plot_hrf.py : sandbox to play with nistats modules. 
- utils.py : some functions collected online useful for general objects handling. 

The Matlab scripts are:

- transition_proba_distrib_generation.m: Script outputting Ideal Observer transition probabilities means, standard deviation and distributions for a large number of subjects (i.e. experiments)
- bernoulli_proba_distrib_generation.m: Script outputting Ideal Observer Bernoulli probabilities means, standard deviation and distributions for a large number of subjects (i.e. experiments)
- generate_transition_sequence.m: generate sequences of transition probabilities-driven stimuli
- generate_bernoulli_sequence.m: generate sequences of Bernoulli-driven stimuli

The data shall be placed in the data/simu folders, after data generation from the Matlab scripts. 
