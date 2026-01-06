# ELPIGNN
  <img width="927" height="625" alt="image" src="https://github.com/user-attachments/assets/89bbb56e-507c-41ce-aa0c-e0fff7876efa" />

# ELPIGNN: Electride Classification with Graph Neural Networks
A deep learning framework for classifying electride materials using the costum design graph neural networks and physics-informed features.

# Overview
ELPIGNN (Electride Physics-Informed Graph Neural Network) is designed for multi-class classification of electride materials. It combines graph neural networks with angle-aware message passing and physics-based descriptors to classify materials into categories of binary electrides.

##  Quick Start
. Installation bash
###  Clone repository 
-- git clone https://github.com/ehsanab550/ELPIGNN.git

###  Install dependencies
- pip install -r requirements.txt
###  Basic Usage
- python
- from ELPIGNN_model import ELPIGNN, BuildConfig

# Project Structure
+ ELPIGNN_model.py              ----->  Core model architecture and graph building

+ PI_electride_features.py      ----->  Physics-informed feature generation

+ ELPIGNN_sample_17class.py     ----->  Example implementation

+ data/                         ----->  Element properties and pre-computed features

# Key Features
~ Graph-based representation of crystal structures

~ Angle-aware message passing for geometric learning

~ Physics-informed features integrating domain knowledge

~ 17-class classification of electride materials

~ Periodic boundary condition handling

# Model Architecture
* The model processes crystal structures as graphs where:

-- Nodes represent atoms with element property embeddings

-- Edges represent atomic neighbors with distance encoding

-- Angle triplets capture geometric relationships

-- Physics-informed features provide global material descriptors

# Data Files
ptable_final_cleaned2.csv - Element properties for node features

PI_electride_featursfinal2.csv - Pre-computed physics-informed features


Contact
For questions and issues, please open an issue on GitHub.
