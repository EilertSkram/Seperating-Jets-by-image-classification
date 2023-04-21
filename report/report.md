# Seperating Jets by image classification
*A DAT255-project by [Andreas Valen](https://github.com/andreasvalen), [Tobias Sagvaag Kristensen](https://github.com/Tobbelobby) and [Eilert Skram](https://github.com/EilertSkram)*

### Abbreviations


  LHC - Large Hadron Collider
  
  ATLAS - asdsadasd
  
  CNN - A convolutional neural network (CNN) is a type of artificial neural network used primarily for image recognition and processing, due to its ability   to recognize patterns in images. [1]
  
  Decision Tree - dasdas
  
  Random Forest - Random Forest is a supervised machine learning algorithm that is used for classification, regression, and other tasks. It is an ensemble learning method that combines multiple decision trees into a single model. Each decision tree in the Random Forest is trained on a random subset of the training data and a random subset of the features. The output of the Random Forest is determined by aggregating the results of all the decision trees in the ensemble, using either the mode (for classification) or the mean (for regression). Random Forest has several advantages over single decision trees, such as reduced overfitting, increased accuracy, and the ability to handle missing values and outliers. [2]



## Introduction


  In this project, we aim to explore the feasibility of using image classification to separate W-boson from general jets. The project uses generated data from the ATLAS project and modern deep learning libraries like FastAI.

  ### What is a Jet?
  
 In particle physics, jets are collimated sprays of particles produced in high-energy collisions, such as those that occur in particle accelerators or cosmic rays interacting with the Earth's atmosphere. Jets arise from the fragmentation and hadronization of partons, which are the constituent quarks and gluons that make up protons, neutrons, and other hadrons.

When two particles collide at high energies, they can create a shower of new particles, including quarks and gluons. These newly created particles can then interact with other particles in the surrounding area, producing more particles and creating a cascade of particle production. The result is a collimated spray of particles known as a jet.

Jets can be observed and studied using particle detectors such as the ones found in particle accelerators. By measuring the energy and momentum of the particles in the jet, physicists can infer information about the properties of the partons that produced the jet and the strong force that governs their interactions.

Jets are important in particle physics because they are a signature of high-energy collisions and provide a way to study the properties of the fundamental particles and their interactions. They also play a crucial role in the search for new particles and phenomena beyond the Standard Model of particle physics, such as the Higgs boson and supersymmetric particles, which may produce distinctive signatures in the form of jets. [3]

[4]
[5]
[6]
  
  ### What is a Boson?
  
  The W-boson is one of the fundamental particles in the Standard Model of particle physics. It is an elementary particle that mediates the weak nuclear force, which is responsible for the radioactive decay of particles, as well as the fusion reactions that power the sun.

The W-boson comes in two varieties, the W+ and the W-. The W+ carries a positive electric charge, while the W- carries a negative electric charge. Both W-bosons have a mass of approximately 80 GeV/c^2 and a lifetime of about 3×10^−25 seconds.

The weak nuclear force is responsible for the transformation of one type of particle into another. For example, the decay of a neutron into a proton, an electron, and an antineutrino is mediated by the exchange of a W- boson. Similarly, the fusion of two protons in the sun to form a deuterium nucleus is mediated by the exchange of a W+ boson.

The discovery of the W-boson was a major triumph of experimental particle physics. The first evidence for the existence of the W-boson came from experiments at CERN in the 1980s, and the discovery was later confirmed by experiments at Fermilab in the United States.

In summary, the W-boson is a fundamental particle that mediates the weak nuclear force and is responsible for the transformation of one type of particle into another. Its discovery was a major milestone in our understanding of the universe at the most fundamental level.
[7] [8] [9] [10]

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
  HDF5 (Hierarchical Data Format version 5) is a data file format designed to store and organize large and complex data structures, commonly used in scientific and engineering applications. HDF5 files can store a wide range of data types, including numerical, text, and image data, and can be easily accessed and manipulated using a variety of programming languages. [11]

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
  [Creating a custom dataset](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/nbs/customdl.ipynb) we have to make our own custom dataset class. This way when the dataloader “asked” for the next image, it would use the function we created and get a ndarray from the hdf5 file instead.

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
This approach finally allowed us to train the model using a pretrained image classifier, and the model varied between 55 and 79% accuracy. Even though the model trained without any major problems, none of the display functions like show_batch and show_results worked. More importantly, other functions like ClassificationInterpretation could not be implemented either.

We created our own “show batch” function and made some minor progress, but in the end it was impossible to subvert all of the challenges using the top level api. 

Custom show batch:
  ![Initial show batch](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/report/figures/cbs2.png)
 
 #### Approach 2.1: Datablock

Using [fastai’s datablock approach](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/nbs/imageblock-with-custom-get-x-function.ipynb)  worked with very little modification, and we could use the hdf5 files without any modification. 
 
Code for creating datablocks:
```

path = '/kaggle/input/jet-images-train-val-test/jet-images_train.hdf5'
classes = ["general", "W-boson"]
    
h5_file = h5py.File(path, 'r')
signal_data = h5_file['signal']
image_data = h5_file['image']

def label_func(x):
    signal = signal_data[int(x)]
    return classes[int(signal)]

def get_items(x):
    l = len(image_data)
    return [str(i) for i in range(l)]

def get_x(x):
    return torch.from_numpy(image_data[int(x)])
dblock = DataBlock(blocks    = (ImageBlock, CategoryBlock),
                   get_items = get_items,
                   get_x = get_x,
                   get_y     = label_func,
                   splitter  = RandomSplitter(),)

```

This is the approach that the rest of the project is built upon.

  

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
Based on the initial exploration of the micro dataset, the ConvNext model architecture, specifically the ``convnext_tiny`` variant, was determined to be the most effective with a baseline accuracy of 80.2%. Thus ConvNext Tiny was chosen as the model for further exploration.

The custom learning rate was found using FastAi's ``lr_find``-method with the suggested functions set as ``minimum, steep, valley, slide``

![learning rate](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/report/figures/learning_rate.png)

For simplicity's sake, valley was chosen, rather than finding an inbetween rate. 



Next, [the model](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/nbs/w-boson-cnn-final%20(1).ipynb) was fine-tuned using the FastAI ``fine_tune`` method over a period of 20 epochs. An early stoppage callback-function with ``patience=4`` and ``min_delta=0.01`` was implemented. In this context, early stoppage refers to the stopping of the model if no improvement of at least 0.01 is observed between the best measured value and the current value after three epochs.
```



learn.fine_tune(
    20, 
    lrs.valley, 
    cbs=EarlyStoppingCallback(monitor='accuracy',min_delta=0.01, patience=4)
)


```

Each epoch took around 2 hours and ended up indicating an accuracy of around 83%

| epoch |	train_loss |	valid_loss |	accuracy |	time|
|-------|------------|--------------|---------|----|
|0 	|0.397031| 	0.390257 	|0.827182 |	2:02:01|
|1 	|0.385442 |	0.397020 	|0.822535 	|2:01:58|
|2 	|0.382048 	|0.390109 	|0.826784 	|2:01:59|
|3 |	0.381820 	|0.402441 	|0.822551 	|2:01:57|
|4| 	0.391494 	|0.381235 |	0.831185 	|2:02:02|


The method ``Learner.validate()`` could not be run due to errors with the dataloaders and the HDF5 datasets, so a workaround was training the model on the training data, then finetuning 1 epoch on the unseen test data. The final model achieved an accuracy of 82,84% on the unseen test data. 

| epoch |	train_loss |	valid_loss |	accuracy |	time|
|-------|------------|--------------|---------|----|
|0 	|0.400126 	|0.386640 |	0.828495 	|09:44|
 
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

Using [this notebook](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/nbs/mean-image-base-line.ipynb) We experiment with using a mean image to classify the jets.

  ![Initial CNN](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/report/figures/jet_batch.png)

The figure displays one of the batches in the early state of the project. At first glance, it looks like the boson images have a common occurring pattern that is different from the general images. In the boson images, we can see that the pixels in the middle are more activated and are appearing in a straight line, while in the general images, the activations are more sparse.

If this is the case, we can produce a mean image for both classes and use this image to predict whether the image is a boson or general. This can be used as a baseline for measuring the performance of the CNN models. The experiment can be found in the file mean-image-baseline.ipynb. To make the image, we used 314,160 samples from the dataset.


  ![Initial CNN](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/report/figures/jet_boson_mean.png)

The mean boson image supports our hypothesis, as we can see that the most active pixels form a line in the center of the image. Before starting to compare the images, we also made a mean general image.


  ![Initial CNN](https://github.com/EilertSkram/Seperating-Jets-by-image-classification/blob/main/report/figures/jet_general_mean.png)


As you can see, there are no major differences, so our expectations for using it for prediction are low.



To calculate the similarities of the vectors, we used two approaches.

#### Cosine Similarity

> Cosine similarity measures the similarity between two vectors of an inner product space. It is measured by the cosine of the angle between two vectors and determines whether two vectors are pointing in roughly the same direction. 

[12]

To implement this, we used the sklearn implementation of cosine with some additional functionality. We used a number of different thresholds to optimize the output. To make this as fast as possible, we used the smallest subset (jet-images_val). The best result was an accuracy of 0.55.

#### Absolute difference

The method calculates the absolute difference between the mean value of the input array and the mean value of two reference arrays, and then makes a prediction based on which reference array has a smaller absolute difference. With this, we got an accuracy of 0.58.

#### Conclusion on mean image experiment

Our initial hypothesis that boson data has low diversity was wrong. If our hypothesis was right, we would have gotten a higher accuracy, but the baseline image predictor is not better than guesswork. This shows the diversity of the data, and we need a more sophisticated way of identifying patterns in the data.


### Results of CNN the model




## Conclusion and discussion

The accuracy achieved seems to indicate that using CNN to classify W-bosons could be a viable approach. The accuracy on both validation and unseen test sets indicates that with smaller sample size, the training can still achieve an accurate result. Despite the good results, the validation methods used in the project are subpar technical workarounds, so the level of confidence regarding the true accuracy is not ideal and not conclusive. Struggles with both datashape and formats have put big constraints on the effectiveness of the project and have highlighted a need for a streamlined integration between FastAi and HDF5 datasets. 

Exploring other CNN libraries, bar FastAi, have the potential to yield better results, but could not be performed or verified during the project. It could also prove beneficial to explore an ensemble of both tabular and image-data.


Lastly the overall conclusion is that the project is considered a success, justified by the indication that CNN is a viable way of classifying Jet-imagedata, thus achieving the main goal. Despite the technical difficulties connected to FastAi and the HDF5-dataset, the project still has managed to show signs that CNN can be a valuable tool when it comes to Jet-classification-tasks. 



## References


  [1]: https://www.arm.com/glossary/convolutional-neural-network
  
  [2]: https://www.ibm.com/topics/random-forest
  
  [3]: https://pdg.lbl.gov/2020/reviews/rpp2020-rev-w-boson.pdf
  
  "Weak Interactions and W Bosons" article from the University of California, Berkeley 
  
  [4]: https://www2.lbl.gov/abc/w/w.html 
  
  "Discovery of the W and Z bosons" article from the CERN Courier
  
  [5]: https://cerncourier.com/a/discovery-of-the-w-and-z-bosons/ 
  
  "The W and Z Bosons" article from Fermilab 
  
  [6]:  https://www.fnal.gov/pub/science/particle-physics/mysteries/wz-bosons.html 
  
  "Jets in Particle Physics" article from the Particle Data Group 
  
  [7]:   https://pdg.lbl.gov/2019/reviews/rpp2019-rev-jets.html 
  
  
  "Jet physics at the LHC" lecture notes from CERN
  
  [8]:  https://home.cern/science/physics/jet-physics-lhc  
  
  
  "Jet Substructure at the Large Hadron Collider: A Review of Recent Advances in Theory and Machine Learning" article from Annual Review of Nuclear and Particle Science 
  
  [9]:  https://www.annualreviews.org/doi/full/10.1146/annurev-nucl-102019-025022 
  
  
  "Jet Physics" lecture notes from the University of Oxford
  
  [10]:  https://www2.physics.ox.ac.uk/sites/default/files/2019-02/Jet_Physics.pdf 
  
  
  The HDF Group, the organization responsible for developing and maintaining HDF5
  
  [11]: https://www.hdfgroup.org/solutions/hdf5/ 

  Getting to Know Your Data 2.4.7 Cosine Similarity 

  [12]: https://www.sciencedirect.com/topics/computer-science/cosine-similarity
  

