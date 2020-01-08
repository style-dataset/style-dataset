# **Interacting with Literary Style through Computational Tools**

*Sarah Sterman, Evey Huang, Vivian Liu, and Eric Paulos*

CHI 2020

[Full paper](https://doi.org/10.1145/3313831.3376730)

In this repository you will find the supplemental materials and the full “Style Similarity Dataset” for this paper. 

************

# Style Similarity Dataset

To develop tools capturing tacit approaches to style, we collected a novel dataset of style judgments. Rather than asking individuals to categorize or label style, we collect judgments of stylistic similarity, using comparisons within triplets of excerpts, as shown to be effective [here](https://homes.cs.washington.edu/~sagarwal/nmds.pdf) and [here](https://arxiv.org/abs/1105.1033). Excerpts are drawn from contemporary fiction, as it is accessible, commonly read, and showcases diverse styles. It is important to note that we would expect to find a great deal of disagreement across individuals in how they judge passages, therefore we collected seven judgments per comparison.

The dataset consists of: 

- *Comparisons:* Crowdworkers read a set of three excerpts of text and compare the style of the first excerpt (A) to the following two (B, C), then judge which of B or C is most stylistically similar to A.
- *Explanations:* Each crowdworker provides a few words of free text to justify their decision, by describing what is similar between A and their choice of B/C.
- *Intensities:* Each crowdworker indicates on a scale of 1-5 how similar their choice of B/C is to A.

For a full description of the creation of the dataset, please see the Dataset section of the paper. Extra details are provided below. 

************

# What’s Here

*Scripts run under Python 2.7 unless otherwise noted.*

## raw data 
- this folder contains compressed csvs for the anonymized crowdsourcing results for the Style Similarity Dataset.
- the numbers in the name of each CSV correspond to the table below.  

## cleaned data 
- this folder contains compressed  csvs containing “good faith” responses, and are the result of running  f8\_clean\_bad\_rows.py  
- the numbers in the name of each CSV correspond to the table below. 

## excerpts and metadata
- this folder contains the excerpts and associated metadata used to generate the corpuses
- corpus numbers correspond to to the table below.
- metadata includes title, author, genres, etc.
- use the "ASIN" column as the id to match metadata to excerpts

## demographic data
- a csv with demographic information about contributors
- the \_worker\_id column matches with the style comparison data csvs

## data processing scripts
### f8\_column\_definitions.py 
- a reference file explaining columns in Figure Eight raw and clean data csvs 

### file\_utilities.py
- a file reading utility script used across several scripts and the model files
	
### f8\_clean\_bad\_rows.py
- script used to clean "bad-faith" responses from collected data.  This script produces the files in the cleaned\_data folder

***********

# Generating Excerpts

In order to crowdsource comparisons of style, we need to create triplets of text excerpts. To generate the excerpts, we retrieved plain text from publicly available previews of fiction published through Amazon Kindle. These books were pulled from seven genre categories as listed by Amazon: Action and Adventure, Contemporary, Historical, Horror, Humor, Literary Fiction, and World Literature. Each corpus (described below) includes texts from all of these genres. Amazon Kindle is used to emphasize contemporary fiction. Other sources, such as Project Gutenberg, emphasize older works in which the conventions of the era may overwhelm more subtle differences in style. We distributed our selection of texts from across the ordering of search results, so as to include texts beyond the top results, by selecting only 3 from every subsequent 10 results on the Amazon search results pages. 


We extracted excerpts of approximately 200 words from each preview. Since the first paragraphs of a book are often quite different from the rest of the text, excerpts were extracted from the middles and ends of the previews. Beginning at the end of the preview's text, we searched for continuous chunks of 200 words that obeyed certain parameters  (see Explaining the Triplet Corpuses for more detail on parameters), for example, not containing dialogue. In this way, we extracted excerpts from similar locations in the source texts, avoiding stylistic confounds that might arise from having excerpts from many narrative points, such as incipts, climaxes, denouements, etc.  We rounded each excerpt to the nearest sentence end above 200 words, so that the excerpts would not end in the middle of a sentence.  If a preview did not have enough valid chunks of text, it was skipped.  


Choosing a style unit of 200 words allows us to analyze prose style at the paragraph level. While choosing a granular unit of comparison means we cannot look at style on the level of narrative structure,
it supports investigating the local style of fragments of text (such as rhythm, sentence structure, vocabulary, etc.). 

2 to 4 excerpts were extracted from each text. Texts were split evenly among genres. 


Since the number of combinations of three excerpts is prohibitively large, we generated a random subset of possible triplets for crowdsourcing. Each excerpt serves as the "anchor" in a triplet a fixed number of times; the anchor refers to excerpt A, against which B and C are compared. To avoid confounds such as shared character names, excerpts from the same text do not occur in the same triplet. 

************

# Explaining the Triplet Corpuses


Triplets are split into 7 groups for crowdsourcing, what we will call the “triplet corpuses”. These different corpus types allow us to investigate a variety of relationships between texts.  For example, corpuses can either include or exclude excerpts with quotations or dialogue, since the presence or absence of dialogue may have an oversized effect on style comparisons. Corpus 4 contains only excerpts with quotations; Corpus 5 contains some excerpts with quotations and some without; all others do not contain quotations. Similarly, there is a group of comparisons with a tight network of connections between excerpts (Corpus 3), where each excerpt acts as an anchor more often among a smaller pool of excerpts. This type of corpus may be useful for clustering algorithms. Corpuses like Corpus 2 are less tightly connected, but provide a larger diversity of texts. 

Each of the corpuses is generated from a disjoint set of texts; i.e. all the excerpts in Corpus 1 come from different books than the excerpts in Corpus 2. The triplets in each corpus are created according to a particular set of rules: we vary the number of excerpts, how many excerpts are pulled from a single text, the number of times an excerpt is used as the anchor in a triplet, and whether or not the excerpts include quotations. 

For example, Corpus 1 draws excerpts from 105 texts, each excerpt will be the anchor excerpt in a triplet 15 times, and excerpts do not have quotations. Each text provides two excerpts, so there are a total of 210 excerpts in Corpus 1, resulting in 3150 total triplets. Two excerpts from the same preview text will never appear in a triplet together, in order to prevent biasing results based on shared character names or other content indicators. For a full characterization, see the  table below. 

The quantity of good faith responses varies between the triplet corpuses.  Corpuses 1, 2, 4 and 5 paid participants immediately for responses, while corpuses 3, 6, and 7 used a "bonusing" approach where participants were paid a small amount up front, and then paid extra if the response was found to be reasonable.  Bonusing seems to have encouraged a higher percentage of good faith responses. 

### Characterization

Columns 2 - 7 are parameters of corpus generation; columns 8 - 11 are results from data collection. For details on Good-Faith Judgments and High-Agreement Triplets, see the paper section "Modeling Style with the Similarity Dataset."

|  Corpus | Texts  | Excerpts per Text  |  Excerpts *(Texts x Excerpts per text)* |  Excerpts include quotes? | Number of times an excerpt is used as an Anchor | Triplets *(Excerpts x Anchors)* | Collected Judgments | Good-Faith Judgments | Good-Faith Yield *(good-faith / collected)* | High-Agreement Triplets |
|---|---|---|---|---|---|---|---|---|---|---|
|1 | 105 | 2 | 210 | None | 15 | 3150 | 21375 | 3741 | 18% | 240 | 
|2 | 315 | 2 | 630 |  None | 5 | 3150  | 22050 | 2985 | 14% | 118 | 
|3 | 28 | 2 | 56 | None | 30 | 1680  | 11775 | 11089 | 94% | 883 | 
|4 | 105 | 2 | 210 |  All | 15 | 3150  | 22050 | 2315 | 10%  | 52 | 
|5 | 35 | 2| 70 | Some | 15 | 1050 | 7320 | 1445 | 20% | 92 | 
|6  | 105 | 2 | 210 | None | 15 | 3150 | 22050 | 18055 | 82% | 1644 | 
|7  | 105 | 4 | 420 | None | 15 | 6300 | 44100 | 26431 | 60% |  2133 | 
|Total  | 798 | -  | 1806 | - | -  | 21,630 | 150,720 | 66,061 | 44% | 5,162 | 

************

# Task Training

At the beginning of the crowdsourcing task, participants were shown an example of a stylistic comparison, and given the following instructions and refresher on some of the concepts that go into literary style:

- *Style:* The "style" of the piece is the way that the author uses words – the author’s word-choice, sentence structure, figurative language, and sentence arrangement. It is NOT the mood or the meaning of the text. Imagine if 3 people wrote the exact same story with the exact same meaning and plot - the differences are the "style" or "feel".
- *What contributes to feel?* This is not a complete list, but you can think about literary techniques such as:
	- sentence structure
	- pace
	- rhythm
	- vocabulary
	- figures of speech
	- point of view
	- tone
	- alliteration, assonance, etc.
	- allusions
(http://teachers.lakesideschool.org/us/english/ErikChristensen/WRITING%20STRATEGIES/LiteraryStyles.htm)
- *What if it’s unclear?* Go with your gut. Some of these comparisons may have multiple good answers, or you may feel that neither of B nor C is very similar to A. You can express the amount of similarity in the ranking question of "how similar is your choice to A."
- *For more information*: More examples and discussion of 'style' or 'feel' can be found at the following link if you need more guidance: https://literaryterms.net/style/

************

# Working with the Dataset

In this paper, our goal was to train a machine learning model to predict style similarity in the form of the comparisons described above.  We used a neural net, and describe the architecture briefly below.  Other approaches to working with the data might include clustering, or analysis of specific stylometric features. If you try something cool, let us know!  

## Training a Predictive Model

We created a binary classifier trained with a binary cross entropy loss function. It takes as input an excerpt triplet, and classifies it into two categories, B or C, indicating which excerpt is most similar to A.
We pre-process each excerpt in the triplet into sequences of characters, sequences of parts of speech (using [Spacy](https://spacy.io/), and sequences of word embeddings (Stanford GloVe). 
These transformations are motivated by features canonically used in stylometric work: character n-grams, syntactic features (which depend on parts of speech), and lexical features (which depend on the words themselves). The neural net then operates on the sequences independently. An LSTM is used for parts of speech, and separate convolutional nets are used for characters and embeddings. After processing, the output vectors are recombined into a single vector of length 48 that represents each excerpt. A modified L2 norm of these vectors is used to calculate the distances between A and B, and A and C, which determines the final classification. 

<img alt="diagram of neural net" src="https://github.com/style-dataset/style-dataset/blob/master/imgs/ml_flow.png" width="400">

We use [Keras](https://keras.io/) to build our model, and an outline of the model code is below: 

    
```

##################### SET UP ############################

# first, preprocess each excert into sequences of parts of speech, word embeddings, and characters

##################### DEFINE INPUTS ############################

# each excerpt, A, B, and C, are treated as separate inputs, and each input has three subparts:
# main --> parts of speech
# aux --> word embeddings
# aux2 --> characters

excerpt_a_main = Input(shape=(x_shape, y_shape)) 
excerpt_a_aux = Input(shape=(1, MAX_SEQUENCE_LENGTH))
excerpt_a_aux_2 = Input(shape=(x_shape_aux_2, y_shape_aux_2))

excerpt_b_main = Input(shape=(x_shape, y_shape)) 
excerpt_b_aux = Input(shape=(1, MAX_SEQUENCE_LENGTH))
excerpt_b_aux_2 = Input(shape=(x_shape_aux_2, y_shape_aux_2))

excerpt_c_main = Input(shape=(x_shape, y_shape)) 
excerpt_c_aux = Input(shape=(1, MAX_SEQUENCE_LENGTH))
excerpt_c_aux_2 = Input(shape=(x_shape_aux_2, y_shape_aux_2))

##################### DEFINE MODEL STRUCTURE ############################

# each preprocessing type is processed by a separate branch:

# LSTM for the parts of speech
main_model = Sequential([ 
    LSTM(16, dropout=0.1, recurrent_dropout=0.1, input_shape=(x_shape, y_shape), return_sequences=True), 
    LSTM(16, dropout=0.1, recurrent_dropout=0.1),
])

# convolutional model for the embeddings
aux_model = Sequential([ 
        Reshape((MAX_SEQUENCE_LENGTH,), input_shape=(1, MAX_SEQUENCE_LENGTH)),
        Embedding(num_words,
                        EMBEDDING_DIM,
                        embeddings_initializer=Constant(embedding_matrix),
                        trainable=False,
                        input_length=MAX_SEQUENCE_LENGTH),
        Conv1D(16, 5, activation='relu'),
        Dropout(0.1),
        MaxPooling1D(5),
        Conv1D(16, 5, activation='relu'),
        Dropout(0.1),
        MaxPooling1D(5),
        Conv1D(16, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(16, activation='relu')
])

#  convolutional model for the characters
aux_model_2 = Sequential([
    Reshape((y_shape_aux_2,x_shape_aux_2), input_shape=(x_shape_aux_2, y_shape_aux_2)),
    Conv1D(16, 5, activation='relu'),
    Dropout(0.1),
    MaxPooling1D(5),
    Conv1D(16, 5, activation='relu'),
    Dropout(0.1),
    GlobalMaxPooling1D(),
    Dense(16, activation='relu')
])


# represent each excerpt as the concatenation of its three branches  
combine_layer = Sequential([
    Merge([main_model, aux_model, aux_model_2], mode='concat')
])

model_a = combine_layer([excerpt_a_main, excerpt_a_aux, excerpt_a_aux_2])
model_b = combine_layer([excerpt_b_main, excerpt_b_aux, excerpt_b_aux_2])
model_c = combine_layer([excerpt_c_main, excerpt_c_aux, excerpt_c_aux_2])

##################### DEFINE DECISION STRUCTURE ############################

def norming_func(vects):
    x, y = vects
    diff = np.power(np.subtract(x,y),2)
    return diff

def subtract_func(vects):
    x, y = vects
    diff = np.subtract(x,y)
    return diff


# compare excerpt A to B, and compare excert A to C
l2_layer_ab = Lambda(norming_func, name="l2_layer_ab")([model_a, model_b])   
l2_layer_ac = Lambda(norming_func, name="l2_layer_ac")([model_a, model_c])   

# generate a single result from the two comparisons
combined_l2 = Lambda(subtract_func, name="subtraction_layer")([l2_layer_ab, l2_layer_ac])

output = Dense(2, activation='softmax', name='judgelayer')(combined_l2)


##################### TRAIN MODEL ############################

# input all preprocessed transformations of the excerpts into the model
tetrad_model = Model(inputs=[excerpt_a_main, excerpt_a_aux, excerpt_a_aux_2, excerpt_b_main, excerpt_b_aux, excerpt_b_aux_2, excerpt_c_main, excerpt_c_aux, excerpt_c_aux_2], outputs=output)

tetrad_model.compile(optimizer='rmsprop',
      loss= 'binary_crossentropy',
      metrics=['accuracy'])

epochs = 70
label_dev_onehot = keras.utils.to_categorical(label_dev, num_classes=num_classes)

tetrad_model.fit(
    x=[stories_training_a, stories_training_a_aux, stories_training_a_aux_2, stories_training_b, stories_training_b_aux, stories_training_b_aux_2, stories_training_c, stories_training_c_aux, stories_training_c_aux_2], 
    y=label_training_onehot, 
    epochs=epochs, 
    validation_data=([stories_dev_a, stories_dev_a_aux, stories_dev_a_aux_2, stories_dev_b, stories_dev_b_aux, stories_dev_b_aux_2, stories_dev_c, stories_dev_c_aux, stories_dev_c_aux_2], label_dev_onehot)
)
 ```   
