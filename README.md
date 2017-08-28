medGAN
=========================================
medGAN is a generative adversarial network for generating multi-label discrete patient records. It can generate both binary and count variables (i.e. medical codes such as diagnosis codes, medication codes or procedure codes).

#### Relevant Publications

medGAN implements the algorithm introduced in the following [paper](https://arxiv.org/abs/1703.06490):

	Generating Multi-label Discrete Patient Records using Generative Adversarial Networks
	Edward Choi, Siddharth Biswal, Bradley Malin, Jon Duke, Walter F. Stewart, Jimeng Sun  
	Machine Learning for Healthcare (MLHC) 2017

#### Code Description

This code trains a generative adversarial network to generate patient records. This work currently can handle patient records that are aggregated over time, hence represented as a matrix where a row corresponds to a patient, and a column to a specific medical code (e.g. diagonsis code, medication code, or procedure code). The value of the matrix could either be binary (i.e. a specific medical code occurred in the longitudinal patient record or not) or count (i.e. how many times a specific medical code occurred in the longitudinal patient record).
	
#### Running GRAM

**STEP 1: Installation**  

1. medGAN was implemented to run on [TensorFlow](https://www.python.org/) 1.2. TensorFlow can be easily installed in Ubuntu as suggested [here](https://www.tensorflow.org/install/install_linux)

2. Download/clone the GRAM code

**Further description coming soon**
