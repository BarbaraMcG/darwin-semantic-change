# Semantic change detection on Darwin's letters

This repository contains code for detecting semantic change from the letters contained in the collection of the Darwin Correspondence Project (https://www.darwinproject.ac.uk/). The code was developed by Barbara McGillivray.


## Initialisation

conda create -n darwin-env python=3.9

conda activate darwin-env

conda install -c conda-forge jupyter pip

pip install spacy==3.5.4 pydantic==1.10.13

pip install numpy==1.24.4

pip install spacy==3.5.4 gensim==4.3.0 smart-open==6.3.0 pathy==0.11.0 scikit-learn==1.2.2 matplotlib lxml seaborn

jupyter notebook
