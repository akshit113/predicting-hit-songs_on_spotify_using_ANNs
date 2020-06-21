### Predicting Hit Songs on Spotify using Deep Feedforward Neural Nets
In this project, you will learn step-by-step about various stages of a applying a machine learning algorithm to a dataset
using various utility functions

### Description of the Dataset
The dataset we will use in this tutorial is the Spotify Hit Songs dataset. This is a dataset consisting of features 
for tracks fetched using Spotify's Web API. The tracks are labeled '1' or '0' ('Hit' or 'Flop') depending on some 
criteria of the author. This dataset can be used to make a classification model that predicts whether a 
track would be a 'Hit' or not. It is a well-understood dataset. All of the variables are continuous and generally
 in the range of 0 to 1. The output variable is binary where 1 is for songs that are hits and 0 for songs that are not.

Although the dataset contains data of more than five decades, in this analysis we have focussed on a smaller subset (one decade) 
of the dataset. You can learn more about this dataset: https://www.kaggle.com/theoverman/the-spotify-hit-predictor-dataset

### Features
- **track**: The Name of the track.
- **artist**: The Name of the Artist.
- **uri**: The resource identifier for the track.
- **danceability**: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. 
- **energy**: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. 
- **key**: The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C?/D?, 2 = D, and so on. If no key was detected, the value is -1.
- **loudness**: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db. 
- **mode**: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
- **speechiness**: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks. 
- **acousticness**: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic. The distribution of values for this feature look like this:
- **instrumentalness**: Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0. The distribution of values for this feature look like this:
- **liveness**: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
- **valence**: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
- **tempo**: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration. 
- **duration_ms**:  The duration of the track in milliseconds.
- **time_signature**: An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure).
- **chorus_hit**: This the the author's best estimate of when the chorus would start for the track. Its the timestamp of the start of the third section of the track. This feature was extracted from the data received by the API call for Audio Analysis of that particular track.
- **sections**: The number of sections the particular track has. This feature was extracted from the data received by the API call for Audio Analysis of that particular track.
- **target**: The target variable for the track. It can be either '0' or '1'. '1' implies that this song has featured in the weekly list (Issued by Billboards) of Hot-100 tracks in that decade at least once and is therefore a 'hit'. '0' Implies that the track is a 'flop'.

## Dependencies
* [python](https://www.python.org/) - Programming Language
* [tensorflow](https://www.tensorflow.org/) - TensorFlow is an open-source machine learning library for research and production
* [keras](https://keras.io/) - Keras is a high-level neural networks API
* [sklearn](http://scikit-learn.org/stable/documentation.html) - Scikit-learn is a free software machine learning library for the Python 
* [numpy](http://www.numpy.org/) - NumPy is the fundamental package for scientific computing
* [pandas](https://pandas.pydata.org/) - Pandas is a software library used for data manipulation and analysis
* [pprint](https://python.readthedocs.io/en/stable/library/pprint.html#module-pprint) - The pprint module provides a capability to “pretty-print” arbitrary Python data structures in a form which can be used as input to the interpreter.


## Steps of Appylying the Machine Algorithm
**Step 1**. Get data using `import_and_clean_data(train_filename, mode='train')` <br/>
 - check if the dataset `has_missing_values(df, colnames)`
 - feature engineering
 - one-hot encoding using `one_hot_encode`
 - normalize your data using using `normalize_columns`
 - convert data to `NumPy` array

**Step 2**. `split_dataset` into train-set and test-set <br/>
 - `train_test_split` splits the data in the ratio of test-size to the total dataset, i.e. test_size = 0.2 implies train_size = 0.8 
 - `get_dummies` creates one-hot vectors of the training and test labels. Alternatively, you can use `to_categorical` function

**Step 3**. Create a Sequential Feedforward Neural Network using `get_model` <br/>
 - be careful when designing the model. Here input_size will the number of columns in x_test (one-hot encoded training set). 
 In this case, it is `16`. Similarly, in output layer, i.e. `output_size` will be equal no. of classes. For example, for binary classification it will be `2`.
 It also must have a `softmax` activation function in order to get model predictions in the form of fuzzy class-probablities
 - `compile` the model using loss function, optimizer, and metric: Here, we have chosen `binary_crossentropy` as the loss function since we are performing classification task. Optimiser is `SGD` and metric is `accuracy`
   
**Step 4**. Fit the data on training set and evaluate the model performance on test_set using `fit_and_evaluate` <br/>
 - `fit` the training data `(x_train, y_train)` in the model using `model.fit` method 
 - `evaluate` the model performance on test set using `model.evaluate` method.
 - print the `test_acc` and `test_loss`. This provides the model performance on test_data
 
**Step 5**. Iteratively tune the model hyperparameters <br/>
- Tune the hyperparameters of the network such as learning rate, activation functions, optimizer, depth of the network (number of hidden layers), width of layers (no. of neurons in each layer), DropOut layers, batch size, epochs, L1 or Ll2 regularization. This is basically the part where the magic happens. So, here's your chance to be creative and artistic!

**Step 6**. `make_predictions` on the test_dataset using `model.predict` method <br/>
- This function will make predictions and convert the output and ground truth (test labels) to 1D numpy array. 

**Step 7**. Calculate the predictive power of your model in `calc_accuracy_using_metrics` method
- This function calculates the predictive score of the model using predicted classes, `y_hat` and ground truth `y`
- Uses various sklearn metrics such as `f1_score`, `precision_score`, `jaccard_score`.

## Hyperparamters
Activation function: `selu`  
Optimizer: `SGD`   
learning rate,`α`: `0.01`  
train-test: `0.8/0.2`  
batch_size: `700`  
epochs: `500`  
loss function: `binary_cross_entropy`

## Results
**Accuracy**: 75.14%  
**Jaccard Score**: 61.97%
**F1 Score**: 
**Precision**: 

## Author
[Akshit Agarwal](https://github.com/123)
