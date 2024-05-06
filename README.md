# Integrating AI and ML in Agriculture and Forestry

- **Overview:**
  In the contemporary landscape of technology, the integration of Artificial Intelligence (AI), specifically within the realm of Machine Learning (ML), has played an important role in agriculture and forestry. The widespread exploration and adoption of ML applications in areas such as automated irrigation systems, agricultural drones for field analysis, and crop monitoring arrangements contribute to superficial benefits in addressing several problems.

- **Challenges and Focus:**
  One area that leads attention is the spread of invasive plant species, such as Eastern Red Cedar or called Juniperus virginiana. For example, the management and control of tree types, growth, and structure continues to be a concern involved in reflecting a real-world setting due to the scale and cost, often considered as an impractical theory-oriented approach.

- **Research Objective:**
  This research explores the integration of automated machine learning (AutoML) and the Machine Learning Operations (MLOps) philosophy, focusing on the automation of ML tasks in real-world applications. As businesses transition from testing to scaling, challenges arise in the development and maintenance of coded and machine-learned components. ML systems have hyperparameters, and the most fundamental task in AutoML is to automatically set these hyperparameters to optimize performance.

- **Approach:**
  The work addresses such a complexity by applying MLOps tools, particularly in understanding the spatial distribution of the Eastern Red Cedar plant in the Great Plains region of the United States, given its negative impacts on the ecosystem and environment. In a cost-effective way, this study develops a pipeline centered around the application of the Haar-Cascade classifier as an ML algorithm by drone-based aerial image acquisition in different seasonal datasets.

- **Streamlining the ML Lifecycle:**
  In order to streamline the ML lifecycle, which encompasses data collection, preprocessing, training, testing, and deployment, activities from development to deployment and monitoring are involved.

- **Conclusion:**
  Beyond immediate goals, this study lays the groundwork for a tangible method in research endeavors, providing a paradigm of flexibility for identifying various tree species.




# Deep Learning Model Process

## Workflow Overview
This research represents a comprehensive workflow from data collection to the development of a CNN model for the detection of Eastern Red Cedar trees.

## Data Collection
- The initial phase involves collecting a diverse dataset, capturing Eastern Red Cedar trees in various conditions such as different seasons, lighting, and backgrounds.

## Model Development
- **Binary Semantic Segmentation:**
  - Each pixel is classified as either an "Eastern Red Cedar tree" or "Background" (None-Eastern Red Cedar tree).
  - A CNN model is developed for this purpose.

# SAM Model

Upon data collection, manual pixel-level labeling is performed using annotation tools like VGG Image Annotator, Labelbox, or SAM. Data augmentation techniques, such as flipping, rotating, and color balance adjustments, may be applied to artificially expand the dataset and enhance the model’s generalization to unseen variations.

SAM ([Source](https://segment-anything.com/)), developed by Meta AI, is an image segmentation model released in April 2023. It accurately identifies the location of specific objects or every object in an image and is open source under the Apache 2.0 license. SAM utilizes various input prompts, specifying what to segment in an image, allowing for a wide range of segmentation tasks without the need for additional training.

## Model Architecture
The SAM model is decoupled into:
1. A one-time image encoder
2. A lightweight mask decoder capable of running in a web browser within a few milliseconds per prompt.

## Features
- Employing a robust image encoder, a prompt encoder, and a lightweight mask decoder, this distinctive architecture facilitates versatile prompting, real-time mask computation, and awareness of ambiguity in segmentation tasks.
- Introduced by the Segment Anything project, the SA-1B dataset features over 1 billion masks on 11 million images. As the largest segmentation dataset to date, it provides SAM with a diverse and large-scale training data source.
- SAM’s data engine accelerates the creation of large-scale datasets, reducing the time and resources required for manual data annotation. This benefit extends to researchers and developers working on their own segmentation tasks.


## Key Components

This project consists of several key components: Data. Model, Objective Functions and Optimization Algorithms

- **Data**: In image data, each example could represent an individual photograph, with features comprising numerical representations of pixel values. In cases where examples share the same number of numerical features, inputs are referred to as fixed-length vectors, and the constant length of these vectors is termed the dimensionality of the data. While fixed-length vectors are convenient, not all data can be easily represented in this format. Images from the Internet, for instance, may vary in resolution and shape, and text data poses challenges due to its varying length.

The abundance of data simplifies tasks, enabling the training of more powerful models and reducing reliance on preconceived assumptions, however, poor data quality or the inclusion of irrelevant features can lead to subpar performance or unintended consequences, such as perpetuating societal biases in predictive models.

- **Model**: The model component refers to the machine learning or statistical models employed in the project. This could include deep learning models, traditional machine learning algorithms, or custom models developed specifically for the task at hand. Most machine learning involves transforming the data in some sense. Deep learning is differentiated from classical approaches principally by the set of powerful models that it focuses on. These models consist of many successive transformations of the data that are chained together top to bottom, thus the name deep learning.

- **Objective Functions**: Objective functions, also known as loss functions or cost functions, are used to quantify the performance of the model. These functions define what the model aims to optimize during training and evaluation, guiding the learning process towards achieving the desired outcomes.

- **Optimization Algorithms**: Optimization algorithms are algorithms used to minimize the objective functions and update the model parameters during training. These algorithms play a crucial role in training the model efficiently and effectively, ensuring convergence to optimal or near-optimal solutions.

Each of these components plays a vital role in the project's success, contributing to the development, training, and evaluation of machine learning models for various tasks.




# Environment Settings

## Operating System and GPU Utilization
- **Operating System:** Pop!_OS ([Source](https://pop.system76.com/))
  - Pop!_OS is a Linux distribution developed by System76, based on Ubuntu and featuring a customized GNOME desktop known as COSMIC.
  - It provides out-of-the-box support for both AMD and Nvidia GPUs.
  - The distribution supports TensorFlow and CUDA without additional configuration.
  - Includes a recovery partition for system refresh.
  - The latest version, Ubuntu 22.04 LTS, was selected for this study.

## Development Tools
- **Editor:** Visual Studio Code (VS Code) ([Source](https://code.visualstudio.com/))
  - VS Code was chosen as the editor due to its compatibility with Linux.
  - Unique features include support for debugging, syntax highlighting, code refactoring, and embedded Git.
 


# Using Hugging Face and Gradio for Interactive Image Segmentation with SAM

## Introduction

This repository demonstrates how to leverage the power of Hugging Face and Gradio to create an interactive image segmentation application using the Segment Anything Model (SAM). SAM is a versatile model capable of segmenting images based on user-defined points. By integrating SAM with Hugging Face's model hosting and Gradio's user interface components, we create a seamless experience for segmenting images with just a few clicks.

## Why Hugging Face ([Source](https://huggingface.co/))?

### Model Hosting and Management

Hugging Face provides an excellent platform for hosting and managing models. By leveraging Hugging Face's infrastructure, we can easily deploy and share our SAM model, making it accessible to anyone via a simple API.

### Extensive Model Repository

With Hugging Face, we have access to a vast repository of pre-trained models, including transformers and vision models. This enables us to experiment with various architectures and leverage state-of-the-art techniques for image segmentation tasks.

## Why Gradio ([Source](https://www.gradio.app/))?

### User-Friendly Interface

Gradio offers a user-friendly interface for building interactive applications with machine learning models. Its intuitive design allows users to upload images and interact with the model through simple UI components like sliders and buttons.

### Rapid Prototyping

Gradio accelerates the development process by providing pre-built UI components that can be easily integrated with machine learning models. This allows us to quickly prototype and iterate on our image segmentation application without spending time on UI development.

## How to Use

To use this application, simply upload an image and click on the area of interest. The model will segment the image based on the selected point, providing instant feedback to the user.

## Getting Started

To get started, clone this repository and follow the setup instructions in the README. Make sure to install the required dependencies and download the SAM checkpoint file.

## Contributions

Contributions are welcome! If you have ideas for improvements or new features, feel free to open an issue or submit a pull request.

