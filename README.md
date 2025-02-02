# Hate-Speech-Detection-LSTM-NLP
This script demonstrates how to build a neural network model using TensorFlow for classifying hate speech and offensive language from text data, with preprocessing and resampling techniques applied to the dataset.
Key Steps:

Data Loading and Preprocessing:

        The dataset is loaded using pandas.
        Unnecessary columns like Unnamed: 0, count, hate_speech, offensive_language, and neither are dropped.
        Tweets are cleaned by removing non-alphabetical characters and extra spaces.

Text Processing:

        Lemmatization: Using the WordNetLemmatizer from nltk, each tweet is lemmatized to reduce words to their base forms.
        Stopwords Removal: Common words like "the", "is", etc., are removed using nltk.corpus.stopwords.

Text Vectorization:

        One-Hot Encoding: Each tweet is transformed into a one-hot encoded representation using tensorflow.keras.preprocessing.text.one_hot.
        Padding: Sequences are padded to ensure they all have the same length of 20 words using tensorflow.keras.preprocessing.sequence.pad_sequences.

Class Imbalance Handling:

        SMOTE: The Synthetic Minority Over-sampling Technique (SMOTE) is applied to balance the dataset by oversampling the minority class.

Model Architecture:

        A Sequential model with the following layers.
        Embedding layer: Converts one-hot encoded words into dense vectors.
        LSTM layers: Long Short-Term Memory layers are used for sequential data processing.
        Dense output layer: A softmax layer with 3 units for classification into 3 classes (hate speech, offensive language, and neither).

 Model Compilation and Training:
 
        The model is compiled using the Adam optimizer and SparseCategoricalCrossentropy loss function.
        The model is trained for 5 epochs with a batch size of 32.

Model Evaluation:

        The model is evaluated on the test data, and the loss and accuracy are printed.
        Predictions are made using the trained model, and the accuracy score of the predictions is calculated.

This approach utilizes LSTM layers to handle the sequence data, and the SMOTE technique helps to address any class imbalance before model training.
