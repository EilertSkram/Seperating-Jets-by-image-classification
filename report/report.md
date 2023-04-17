# Seperating Jets by image classification
*A DAT255-project by [Andreas Valen](https://github.com/andreasvalen), [Tobias Sagvaag Kristensen](https://github.com/Tobbelobby) and [Eilert Skram](https://github.com/EilertSkram)*

### Abbreviations

<details>
  <summary>Expand</summary>

  LHC - Large Hadron Collider
  
  ATLAS - asdsadasd
  
  CNN - A convolutional neural network (CNN) is a type of artificial neural network used primarily for image recognition and processing, due to its ability   to recognize patterns in images. [1]
  
  Decision Tree - dasdas
  
  Random Forest - adsasda [2]

</details> 

## Introduction

<details>
  <summary>Expand</summary>
  
  In this chapter, we will explore three important topics: jets, bosons, and the deccription and overall goal of this project.

  ### What is a Jet?
  
 In particle physics, jets are collimated sprays of particles produced in high-energy collisions, such as those that occur in particle accelerators or cosmic rays interacting with the Earth's atmosphere. Jets arise from the fragmentation and hadronization of partons, which are the constituent quarks and gluons that make up protons, neutrons, and other hadrons.

When two particles collide at high energies, they can create a shower of new particles, including quarks and gluons. These newly created particles can then interact with other particles in the surrounding area, producing more particles and creating a cascade of particle production. The result is a collimated spray of particles known as a jet.

Jets can be observed and studied using particle detectors such as the ones found in particle accelerators. By measuring the energy and momentum of the particles in the jet, physicists can infer information about the properties of the partons that produced the jet and the strong force that governs their interactions.

Jets are important in particle physics because they are a signature of high-energy collisions and provide a way to study the properties of the fundamental particles and their interactions. They also play a crucial role in the search for new particles and phenomena beyond the Standard Model of particle physics, such as the Higgs boson and supersymmetric particles, which may produce distinctive signatures in the form of jets.
  
  ### What is a Boson?
  
  The W-boson is one of the fundamental particles in the Standard Model of particle physics. It is an elementary particle that mediates the weak nuclear force, which is responsible for the radioactive decay of particles, as well as the fusion reactions that power the sun.

The W-boson comes in two varieties, the W+ and the W-. The W+ carries a positive electric charge, while the W- carries a negative electric charge. Both W-bosons have a mass of approximately 80 GeV/c^2 and a lifetime of about 3×10^−25 seconds.

The weak nuclear force is responsible for the transformation of one type of particle into another. For example, the decay of a neutron into a proton, an electron, and an antineutrino is mediated by the exchange of a W- boson. Similarly, the fusion of two protons in the sun to form a deuterium nucleus is mediated by the exchange of a W+ boson.

The discovery of the W-boson was a major triumph of experimental particle physics. The first evidence for the existence of the W-boson came from experiments at CERN in the 1980s, and the discovery was later confirmed by experiments at Fermilab in the United States.

In summary, the W-boson is a fundamental particle that mediates the weak nuclear force and is responsible for the transformation of one type of particle into another. Its discovery was a major milestone in our understanding of the universe at the most fundamental level.

  ### Project description

  Proton-proton collisions within the ATLAS experiment at The Large Hadron Collider (LHC)
  produce multiple jets. Some of the jets appear more frequent, it is important to separate the
  jets, as the ongoing research for finding new particles often look for specific jets. In addition,
  current and future collision conditions at the LHC produce a large number of less interesting
  jets, which need to be separated from the other jets.

  In this case study we will focus on classifying W-bosons, quarks, and gluons. And if time,
  broaden the scope to other particles. The dataset provided contains pictures of 2D
  representations of energy deposition from particles interacting with a colorimeter. The aim is to
  explore different models, architecture, and deep learning techniques to optimism the result.
  Convolutional neural network has been successfully applied to this task within the ATLAS
  collaboration and can be a natural starting point.

  ### Goals

  - Classifying the different jets
  - Experiment with different models and architectures.
  - If there is time, expand to other particles.
  
 

</details> 


## Data

<details>
  <summary>Expand</summary>
  
  ### HDF5
  HDF5 (Hierarchical Data Format version 5) is a data file format designed to store and organize large and complex data structures, commonly used in scientific and engineering applications. HDF5 files can store a wide range of data types, including numerical, text, and image data, and can be easily accessed and manipulated using a variety of programming languages.

### Label
The structure of the dataset we are working with is tabular data stored in tables.
 The target label is the table/column named signal. Signal is a binary column. 1 indicates a w-boson was found, 0 indicates general jet. 
  ![Initial CNN](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/report/figures/hdf5_tables.png)

### Images
All tables in the dataset, except for the image data, consist of a single column. The image data differs in that it is made up of 25 tables with 25 rows with a single float number in each cell, forming a 25x25 pixel single channel image. The row at the same indices in each table sequentially relates to a row of pixels of the same image where the first table contains the top row, and the 25th table contains the last. This is the same as the NumPy array format. The images are generated from energy sensors.
![InitialCNN](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/report/figures/hdf5_image.png)

### Jet data
jet_pt: "Transverse momentum", meaning the magnitude of the momentum that the jet has across the beam direction. This is a proxy for the energy of the jet.

jet_eta:  A measure of how far the jet is from the beam direction. Eta=0 corresponds to 90 degrees:  Almost all processes will be symmetric around eta=0. See the precise definition here: https://en.wikipedia.org/wiki/Pseudorapidity. 

jet_phi: Angle that measures the position in the plane perpendicular to the beam direction. Here, there should be complete rotational symmetry.

jet_mass: A measure of the mass of the object that created the jet. If the W decays to only quarks, the jet_mass is expected to be equal to the W mass. For QCD jets (like those found in events with label 0), there is no known expected value, but the value is typically much smaller than in W events.

jet_delta_R: Angular distance (actually sqrt(delta_eta^2 + delta_phi^2)) between the two most energetic jets in the event. 0 if there is no more than one jet. 

tau-variables: Substructure within the jet, meaning that the jet is actually a collection of several narrower jets. More details here: https://arxiv.org/pdf/1011.2268.pdf.
  
### Dividing the dataset
Because of the data being structured in tables, the dividing of the dataset is not strictly straight forward. The difficulty increases when we take into account the wish to maintain the 50/50 signal distribution throughout the subsets. Here are the distributions:
 ![Initial CNN](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/report/figures/signal_distribution.PNG)

[Script used for splitting dataset](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/nbs/dividing-the-hdf5-dataset-into-train-test-and-val.ipynb)



</details> 
 

## Approach 

<details>
  <summary>Expand</summary>
  
  ### Initial Plan
  
  
  ### Custom HDF5 Dataset
  Creating a custom HDF5 dataset can be a complex and time-consuming task, requiring careful planning and attention to detail. While HDF5 is a powerful and flexible data format, it can also be difficult to work with, particularly when dealing with large or complex data structures. HDF5 and FastAi lack a good integration. Due to the timeframe of the project, the custom dataset was scrapped after it proved time-consuming. 
  
  ### Converter
  A temporary fix initially was running a script in Kaggle to convert the pictures to PNG, using PIL Image. This was a slow process, but yielded good results.
  
  
  #### CNN
  Variation 1: Converting to PNG
  ![Initial CNN](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/report/figures/init_cnn.png)
  
  Variation 2: Creating custom dataset
  ![Initial custom dataset CNN](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/report/figures/init_cstm_cnn.png)
  
  #### Ensemble
  ![Initial ensemble](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/report/figures/init_ens.png)
  
  
</details> 


## Model

<details>
  <summary>Expand</summary>
  
  ### Initial Model Exploration
  
  #### Baseline CNN
  Resnet-18 was used as the baseline model, using accuracy as metric for measurement. 
  `learn = vision_learner(dls,arch="resnet18",  metrics=accuracy)`
  
  
  <img width="571" alt="image" src="https://user-images.githubusercontent.com/54356437/231675841-9632fadf-7e35-4281-be16-6788f8f27e6d.png">

  Using learning rate of 10^(-3) 
  
  <img width="336" alt="image" src="https://user-images.githubusercontent.com/54356437/231675564-f959ef10-8e31-4111-aaa1-ff7797c00a8d.png">

  The baseline model yielded an accuracy of 82,189%
  
  #### Baseline model for tabular data
  
  #### Baseline ensemble

  
</details> 

## Conclusion and discussion
<details>
  <summary>Expand</summary>
  dasdasdaslkdjada
</details> 

## References

<details>
  <summary>Expand</summary>

  [1]: https://www.arm.com/glossary/convolutional-neural-network
  [2]: https://www.ibm.com/topics/random-forest
  
      "W and Z bosons" article from the Particle Data Group: https://pdg.lbl.gov/2020/reviews/rpp2020-rev-w-boson.pdf
    "Weak Interactions and W Bosons" article from the University of California, Berkeley: https://www2.lbl.gov/abc/w/w.html
    "Discovery of the W and Z bosons" article from the CERN Courier: https://cerncourier.com/a/discovery-of-the-w-and-z-bosons/
    "The W and Z Bosons" article from Fermilab: https://www.fnal.gov/pub/science/particle-physics/mysteries/wz-bosons.html
  
      "Jets in Particle Physics" article from the Particle Data Group: https://pdg.lbl.gov/2019/reviews/rpp2019-rev-jets.html
    "Jet physics at the LHC" lecture notes from CERN: https://home.cern/science/physics/jet-physics-lhc
    "Jet Substructure at the Large Hadron Collider: A Review of Recent Advances in Theory and Machine Learning" article from Annual Review of Nuclear and Particle Science: https://www.annualreviews.org/doi/full/10.1146/annurev-nucl-102019-025022
    "Jet Physics" lecture notes from the University of Oxford: https://www2.physics.ox.ac.uk/sites/default/files/2019-02/Jet_Physics.pdf
  
  The HDF Group, the organization responsible for developing and maintaining HDF5: https://www.hdfgroup.org/solutions/hdf5/
  
</details> 
