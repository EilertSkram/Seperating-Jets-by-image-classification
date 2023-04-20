# Seperating Jets by image classification
*A DAT255-project by [Andreas Valen](https://github.com/andreasvalen), [Tobias Sagvaag Kristensen](https://github.com/Tobbelobby) and [Eilert Skram](https://github.com/EilertSkram)*

### Abbreviations


  LHC - Large Hadron Collider
  
  ATLAS - asdsadasd
  
  CNN - A convolutional neural network (CNN) is a type of artificial neural network used primarily for image recognition and processing, due to its ability   to recognize patterns in images. [1]
  
  Decision Tree - dasdas
  
  Random Forest - adsasda [2]



## Introduction


  
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
  
 


## Data

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

 

## Approach 

  
<details>
  
### Initial Plan
Initially we wanted to compare the performance of two models trained on the tabular data and the images respectively.

#### Random forest
We picked random forest at random, and created a model that got close to 80% accuracy. Notably, to be able to use the sklearn’s top level api we had to flatten the tables into a single tabular datafile.

#### Fastai image classifier
We got completely stuck when trying to create the image classifier. There seemed to be a compatibility issue as all of the top level api’s expected the images to be in separate files, and to be given a list of paths to the files. 

### Divide and conquer
 There was little documentation to help with our specific problem, however, we found some promising work of people creating custom datasets from hdf5 files. This custom dataset approach was not directly applicable to our situation, but it was close enough to warrant taking a closer look. 
As there was no guarantee it would lead anywhere, we divided the workflow and explored two different avenues at the same time: custom dataset, and converting the numpy array to png.
 
  
  #### Converter
  Approach 1: Converting to PNG
  ![Initial CNN](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/report/figures/init_cnn.png)
We created a Kaggle [notebook to convert the pictures to PNG](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/nbs/boson-convert-manual.ipynb), using PIL Image. It used a hdf5 file as input, and saved 100.000 images in kaggle’s output folder. The path to the output folder of that notebook was then used by a dataloader in a different notebook. This was a slow process, but ultimately worked and yielded results in the same realm as the random forest. 

Converting back and forth was not ideal as some precision might be lost in the conversion, and the dataset size limitations imposed by Kaggle meant we could only use a subset of our data at the time. This approach was ultimately scrapped as we found success in the other approach.

  #### Custom HDF5 Dataset Class
Approach 2: Creating a custom dataset
  ![Initial custom dataset CNN](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/report/figures/init_cstm_cnn.png)
  Creating a custom HDF5 INSERT LINK TO CUSTOM dataset we have to make our own custom dataset class. This way when the dataloader “asked” for the next image, it would use the function we created and get a ndarray from the hdf5 file instead.

Function from the dataset class:
```    
      def __getitem__(self, idx):
        if self.data is None:
            _data = h5py.File(self.file_path, 'r')['image']
            self.data = _data
        if self.label is None:
            self.label = h5py.File(self.file_path, 'r')['signal']
            
        image = self.covertFromNdarrayToTensorRGB(self.data[idx])
        label = int(self.label[idx]) 
        return image, label
  ```

</details>
  
  #### CNN
  Variation 1: Converting to PNG
  ![Initial CNN](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/report/figures/init_cnn.png)
  
  Variation 2: Creating custom dataset
  ![Initial custom dataset CNN](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/report/figures/init_cstm_cnn.png)
  
  #### Ensemble
  ![Initial ensemble](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/report/figures/init_ens.png)
  

 ## CNN

 
### Model Exploration
Due to Kaggle’s limited computing power, an active choice to explore less demanding models was made. The models chosen were ConvNext and Resnet. The micro dataset was used, containing 10% of the overall images, totalling 87266 images with a 50/50 split of W-boson and general jets. 

The exploration was done by resizing the images from 25x25 pixels to 64x64 pixels, meeting the minimum requirement of ConvnNext. The resizing was done via FastAI’s item transformation-method as ``item_tfms=[Resize(64, method='squish')]``.

Pre-trained models were used, setting fine tuning for 5 epochs, then repeat the process with varying levels of architectural complexity.

|  Model | Accuracy after 5 epochs |
|--------- |--------- |
| Resnet18 | 78% |
| Resnet26 | 75% |
| Resnet59 | 69% |
| ConvNext_tiny | 80,2% |
| ConvNext_small | 80,1% |
| ConvNext_large | 79,6%|


Both architectures performed reasonably well. However, attempting to increase the complexity of the architecture by adding additional convolutional layers, we found that the accuracy of the model actually decreased. In fact, we observed a consistent trend where the accuracy decreased as the complexity of the model increased, indicating that a more complex architecture was not necessarily better. The trend was hypothesized to be connected to the simplistic and small image size, meaning additional convolutions wouldn't necessarily add information gain.

### Fine Tuning CNN
Based on the initial exploration of the micro dataset, the ConvNext model architecture, specifically the ``convnext_tiny`` variant, was determined to be the most effective with a baseline accuracy of 80.2%.

To determine an optimal learning rate for the model, the FastAI method ``lr_find`` was utilized. For simplicity, ``lr.valley`` was selected.

Next, the model was fine-tuned using the FastAI ``fine_tune`` method over a period of 20 epochs. An early stoppage callback-function with ``patience=3`` and ``min_delta=0.01`` was implemented. In this context, early stoppage refers to the stopping of the model if no improvement of at least 0.01 is observed between the best measured value and the current value after three epochs.

The fine-tuning process was conducted on the micro/training dataset. The final model achieved an accuracy of Y after X epochs.

 
 
## Tabular Data - Decision Trees

To verify that using CNN for classifying Jets was a competitive approach, variations of decision trees were created for the tabular data in the Jet-dataset. Similar to the CNN-baseline model, little to no tweaks to the dataset was done. The task is a binary classification problem, so F1 score was used as the metric of performance regarding accuracy and recall.

 

One decision tree and three decision tree ensembles were run, mainly relying on the SKLearn library. Cross validation was used with the fold variable as ``cv = 10``, meaning the data will be divided into 10 equal parts and the model will be trained and evaluated 10 times, with each part used as the validation data once. The mean and standard deviation of the scores for each model was then calculated and printed. 

 
|  Model | F1 Score Mean | F1 Score Standard deviation|
|--------- |--------- | ------ |
| XGBoost | 0.813 | 0.00096 |
| SKLearn Gradient Boost | 0.809 | 0.00104985 |
| SKLearn Random Forest | 0.8104 | 0.0014667 |
| SKLearn Decision Tree | 0.732 | 0.001601 |


XGBoost and Random Forest performed the best out of the four, resulting in F1 scores of 81,3 and 81,0 respectively, thus being chosen as the tabular data-models for the CNN-model to compete with.




  
  ## Baseline mean image

![Initial CNN](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/report/figures/jet_batch.png)

The figure displays one of the batches in the early state of the project. At first glance, it looks like the boson images have a common occurring pattern that is different from the general images. In the boson images, we can see that the pixels in the middle are more activated and are appearing in a straight line, while in the general images, the activations are more sparse.

If this is the case, we can produce a mean image for both classes and use this image to predict whether the image is a boson or general. This can be used as a baseline for measuring the performance of the CNN models. The experiment can be found in the file mean-image-baseline.ipynb. To make the image, we used 314,160 samples from the dataset.


  ![Initial CNN](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/report/figures/jet_boson_mean.png)

The mean boson image supports our hypothesis, as we can see that the most active pixels form a line in the center of the image. Before starting to compare the images, we also made a mean general image.


  ![Initial CNN](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/report/figures/jet_general_mean.png)


As you can see, there are no major differences, so our expectations for using it for prediction are low.



To calculate the similarities of the vectors, we used two approaches.

#### Cosine Similarity

> Cosine similarity measures the similarity between two vectors of an inner product space. It is measured by the cosine of the angle between two vectors and determines whether two vectors are pointing in roughly the same direction. (source: https://www.sciencedirect.com/topics/computer-science/cosine-similarity)

To implement this, we used the sklearn implementation of cosine with some additional functionality. We used a number of different thresholds to optimize the output. To make this as fast as possible, we used the smallest subset (jet-images_val). The best result was an accuracy of 0.55.

#### Absolute difference

The method calculates the absolute difference between the mean value of the input array and the mean value of two reference arrays, and then makes a prediction based on which reference array has a smaller absolute difference. With this, we got an accuracy of 0.58.

#### Conclusion

Our initial hypothesis that boson data has low diversity was wrong. If our hypothesis was right, we would have gotten a higher accuracy, but the baseline image predictor is not better than guesswork. This shows the diversity of the data, and we need a more sophisticated way of identifying patterns in the data.




## Conclusion and discussion


## References


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
  

