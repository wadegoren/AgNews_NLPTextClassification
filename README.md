# Text Classification with Deep Learning using AG News Dataset
Overview
This repository contains the code for a text classification project using deep learning techniques. The goal of the project is to categorize news articles into different topics using the AG News dataset. The implementation utilizes a convolutional neural network (CNN) with pre-trained word embeddings to capture the context of words effectively.

Dataset
The model is trained and evaluated on the AG News Classification Dataset, which consists of news articles categorized into four classes: World, Sports, Business, and Sci/Tech. The dataset is loaded from CSV files (train.csv and test.csv) available in the ag_news_dataset directory.

Features
Model Architecture:

The implemented model is a convolutional neural network (CNN) with multiple layers, including 1D convolutional layers, batch normalization, and global max-pooling.
Dropout layers are added to address overfitting.
The model is trained using the categorical crossentropy loss function and the Adam optimizer.
Data Preprocessing:

Text data preprocessing is performed using the TensorFlow TextVectorization layer, converting words into numeric vector representations.
The GloVe pre-trained word embeddings (300-dimensional) are used to enhance the model's understanding of word contexts.
Training and Evaluation:

The model is trained on the training set, and its performance is evaluated on the test set.
A learning rate scheduler is employed to optimize training efficiency.
Confusion matrix analysis is performed to assess the model's accuracy and identify potential areas for improvement.
Usage
Download the Dataset:

Download the AG News dataset from Kaggle and place the CSV files in the ag_news_dataset directory.
Download GloVe Embeddings:

Run the provided commands in the code to download the GloVe pre-trained word embeddings.
Install Dependencies:

Make sure to install the required libraries mentioned in the code using pip install.
Run the Code:

Execute the provided Python script in your preferred environment, ensuring compatibility with TensorFlow and other dependencies.
Explore Variations:

Consider experimenting with hyperparameters, model architecture, or using different pre-trained word embeddings to explore variations and potentially improve the model's performance.
Results
After training the model, the test accuracy achieved is approximately 89.54%. The confusion matrix provides insights into the model's performance across different classes.

References
Base Code: Keras Examples
Research Paper: Link to relevant research paper
Feel free to explore, modify, and contribute to enhance the text classification model. If you encounter any issues or have suggestions for improvements, please create an issue or submit a pull request. Happy coding!
