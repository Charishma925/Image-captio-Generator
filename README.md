# Image Caption Generator

This project demonstrates how to build an **Image Caption Generator** using Deep Learning techniques, specifically by combining **CNN** (Convolutional Neural Network) and **RNN** (Recurrent Neural Network with LSTM units).

## Project Overview

The goal of the project is to automatically generate captions for images using a combination of **Computer Vision** and **Natural Language Processing (NLP)**. The model is capable of interpreting an image and producing human-like captions.

## Platform

This project was developed using the **Kaggle Notebook environment**, leveraging GPU resources for model training and testing.

## Dataset

We used the **Flickr8k dataset**, which was obtained from [Kaggle Datasets](https://www.kaggle.com/code/charishma09cherry/image-caption-generator/input). The dataset contains:
- 8,000+ images of various real-world scenes.
- 5 captions per image, describing the visual content in natural language.

## Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy, Matplotlib**
- **NLTK (for text preprocessing)**
- **VGG16** (pretrained CNN for feature extraction)
- **LSTM** (for sequential caption generation)
- **BLEU Score** (for evaluating caption quality)

## Model Architecture

- **CNN (VGG16)** extracts feature vectors from images.
- **Tokenized captions** are passed through an **embedding layer**.
- **LSTM** receives both visual features and textual input to generate coherent captions.
- The model is trained using a custom data generator and optimized using the **Adam optimizer**.

## Evaluation

The performance of the model was measured using the **BLEU Score**:
- Achieved a BLEU-1 score of approximately **0.683**.

## Output Examples

The model successfully generates captions like:
- *"a group of people standing around a horse"*
- *"a dog is running through the grass"*

These captions closely match human descriptions and are generated based on unseen test images.

## How to Run

To reproduce the results:
1. Open the [Kaggle Notebook](https://www.kaggle.com/code/charishma09cherry/image-caption-generator/notebook).
2. Import the Flickr8k dataset from [Kaggle Datasets](https://www.kaggle.com/code/charishma09cherry/image-caption-generator/input).
3. Run the full pipeline — from data preprocessing to model training and caption generation.



---


