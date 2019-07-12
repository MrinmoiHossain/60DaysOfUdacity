![alt text](./ChallengePoster.jpg)

# Secure and Private AI Scholarship Challenge Nanodegree Program

This [GitHub](https://github.com/) repository contains the course material related to Udacity's [Secure and Private AI Scholarship Challenge Nanodegree Program](https://classroom.udacity.com/nanodegrees/nd185/).

| Program Start | Challenge Start | Challenge End  |   Program End   |
| ------------- | --------------- | -------------- | --------------- |
|  May 30, 2019 |  June 27, 2019  | August26, 2019 | August 30, 2019 |


# Core Curriculum
### Lesson - 01 : Welcome to the Scholarship Challenge

### Lesson - 02 : Deep Learning with PyTorch
* **Part 1:** Introduction to PyTorch and using tensors
* **Part 2:** Building fully-connected neural networks with PyTorch
* **Part 3:** How to train a fully-connected network with backpropagation on MNIST
* **Part 4:** Exercise - train a neural network on Fashion-MNIST
* **Part 5:** Using a trained network for making predictions and validating networks
* **Part 6:** How to save and load trained models
* **Part 7:** Load image data with torchvision, also data augmentation
* **Part 8:** How to accelerate network computations using a GPU
* **Part 9:** Use transfer learning to train a state-of-the-art image classifier for dogs and cats

### Lesson - 03 : Introducing Differential Privacy
### Lesson - 04 : Evaluating the Privacy of a Function
### Lesson - 05 : Introducing Local and Global Differentail Privacy
### Lesson - 06 : Differentail Privacy for Deep Learning
* **Part 1:** Differentail Privacy for Deep Learning
* **Part 2:** Example Scenario of Deep Learning in a Hospital
* **Part 3:** Generating Differentailly Private Labels for a Dataset
* **Part 4:** PATE Analysis
* **Part 5:** Where to Go From Here
* **Part 6:** Final Project
* **Part 7:** Guest Interview

### Lesson - 07 : Federated Learning
* **Part 1:** Introduction of Federated Learning
* **Part 2:** Introduction & Installing PySyft and Introduction OpendMined
* **Part 3:** Basic Remote Execution in PySyft
* **Part 4:** Playing with Remote Tensors
* **Part 5:** Introducing Remote Arithmetic in PySyft
* **Part 6:** Simple Linear Model
* **Part 7:** Garbage Collection and Common Errors
* **Part 8:** Toy Model in Federated Learning
* **Part 9:** Advanced Remote Execution Tools
* **Part 10:** Pointer Chain Operations
* **Part 11:** Final Project Description
* **Part 12:** Guest Interview

### Lesson - 08 : Securing Federated Learning
### Lesson - 09 : Encrypted Deep Learning
### Lesson - 10 : Challenge Course Wrap Up




# Dependencies
- All necessary needed dependencies are well documented in [udacity/deep-learning-v2-pytorch](https://github.com/udacity/deep-learning-v2-pytorch#dependencies) 

1. Install Conda
- Download latest version of Anaconda from [Anaconda Website](https://www.anaconda.com/distribution/)

2. Install PyTorch and torchvision; this should install the latest version of PyTorch

- __Linux__ or __Mac__: 
```
conda install pytorch torchvision -c pytorch 
```
- __Windows__: 
```
conda install pytorch -c pytorch
pip install torchvision
```

3. Install a few required pip packages
```
pip install opencv-python jupyter matplotlib pandas numpy pillow scipy tqdm scikit-learn scikit-image seaborn h5py ipykernel bokeh pickleshare
```

4. Install PySyft packages
```
conda create -n pysyft python=3
conda activate pysyft
pip install syft
```
