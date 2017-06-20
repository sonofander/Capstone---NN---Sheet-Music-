# Sheet Music Classification
#### Neural Network Project for Galvanize's Data Science Immersive

Music is categorized into twelve major keys that allows musicians to follow a rough framework when playing in a particular key. These keys are usually determined from the key signature at the beginning of the stanza. The focus of this project was to classify sheet music into their respective keys.

![alt text]( https://method-behind-the-music.com/theory/images/fifths2-91a196f1.png "Circle of Fifths")

#### Data Collection

The images were collected from over 4000 pdfs using Beautiful Soup to collect:
* Key signature
* Time signature
* Range of keys
* Tempo
* Song complexity/difficulty
* Link to pdf

Beautiful Soup with the html parser and regular expressions allowed for extraction of the song's attributes.
> MusicScraping.ipynb

#### Preprocessing

Preprocessing was used to convert the pdf images into uniform images through image reduction, quality checks, creating similar png images, and ensuring all gray-scale values were between 0 and 1. The majority of these processes were performed with bash scripts to process all images from the chosen keys.
> convert ./"$f" -density 600 -crop 300x300+0+0 -quality 100 ./${f%.pdf}.png

A pandas dataframe was created for future analysis of the attributes and allowed for different classifications instead of key to be linked to each song.

####

#### Neural Network Parameter Selection

Hyperopt or hyperparameter optimizer

#### Challenges

* Key changes during the piece
* Class Imbalances
* Minor keys
* Activation function Selection
* Setting up AWS GPU server to run parameter Selection


#### Final Model Results
After implementing two convolutional layers, a dense layer, and a softmax output with 12 predictions, accuracy improved to 82% accuracy, which compares to the 8.33% chance of guessing. I

#### Future Implementation/Features
* Improve accuracy with more images
* Able to classify by difficulty, minor keys, the key range, time signature, and genre
* Similarity comparison website to recommend next piece of music to play
