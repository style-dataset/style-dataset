
In this repository you will find the supplemental materials for the paper submission “Interacting with Literary Style through Computational Tools”, including the full “Style Similarity Dataset”. 

The scripts run under Python 2.7 unless otherwise noted.  

************

# What’s Here


## raw data compressed
- this folder contains compressed csvs for the anonymized crowdsourcing results for the Style Similarity Dataset.
- the numbers in the name of each CSV correspond to the table below.  

## cleaned data compressed
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

## data procssing scripts
### f8\_column\_definitions.py 
- a reference file explaining columns in Figure Eight raw and clean data csvs 

### file\_utilities.py
- a file reading utility script used across several scripts and the model files
	
### f8\_clean\_bad\_rows.py
- script used to clean "bad-faith" responses from collected data.  This script produces the files in the cleaned\_data folder


************

# Explaining the Triplet Corpuses

Excerpts are combined into triplets (A, B, C).  The A excerpt is the anchor; a rater compares B and C to A, and chooses which is most stylistically similar to A.

These triplets are split into 7 groups, what we will call the “triplet corpuses”. These different corpus types allow us to investigate a variety of relationships between texts.  For example, corpuses can either include or exclude excerpts with quotations or dialogue, since the presence or absence of dialogue may have an oversized effect on style comparisons. Corpus 4 contains only excerpts with quotations; Corpus 5 contains some excerpts with quotations and some without; all others do not contain quotations. Similarly, there is a group of comparisons with a tight network of connections between excerpts (Corpus 3), where each excerpt acts as an anchor more often among a smaller pool of excerpts. This type of corpus may be useful for clustering algorithms. Corpuses like Corpus 2 are less tightly connected, but provide a larger diversity of texts. 

Each of the corpuses is generated from a disjoint set of texts; i.e. all the excerpts in Corpus 1 come from different books than the excerpts in Corpus 2. The triplets in each corpus are created according to a particular set of rules: we vary the number of excerpts, how many excerpts are pulled from a single text, the number of times an excerpt is used as the anchor in a triplet, and whether or not the excerpts include quotations. 

For example, Corpus 1 draws excerpts from 105 texts, each excerpt will be the anchor excerpt in a triplet 15 times, and excerpts do not have quotations. Each text provides two excerpts, so there are a total of 210 excerpts in Corpus 1, resulting in 3150 total triplets. Two excerpts from the same preview text will never appear in a triplet together, in order to prevent biasing results based on shared character names or other content indicators. For a full characterization, see the above table. 

The quantity of good faith responses varies between the triplet corpuses.  Corpuses 1, 2, 4 and 5 paid participants immediately for responses, while corpuses 3, 6, and 7 used a "bonusing" approach where participants were paid a small amount up front, and then paid extra if the response was found to be reasonable.  Bonusing seems to have encouraged a higher percentage of good faith responses. 

|  Corpus | Texts  | Excerpts per Text  |  Excerpts |  Excerpts include quotes? | Anchors | Triplets | Collected Judgments | Good-Faith Judgments | Good-Faith Yield | High-Agreement Triplets |
|---|---|---|---|---|---|---|---|---|---|---|
|1 | 105 | 2 | 210 | None | 15 | 3150 | 21375 | 3741 | 18% | 240 | 
|2 | 315 | 2 | 630 |  None | 5 | 3150  | 22050 | 2985 | 14% | 118 | 
|3 | 28 | 2 | 56 | None | 30 | 1680  | 11775 | 11089 | 94% | 883 | 
|4 | 105 | 2 | 210 |  All | 15 | 3150  | 22050 | 2315 | 10%  | 52 | 
|5 | 35 | 2| 70 | Some | 15 | 1050 | 7320 | 1445 | 20% | 92 | 
|6  | 105 | 2 | 210 | None | 15 | 3150 | 22050 | 18055 | 82% | 1644 | 
|7  | 105 | 4 | 420 | None | 15 | 6300 | 44100 | 26431 | 60% |  2133 | 
|Total  | 798 | -  | 1806 | - | -  | 21,630 | 150,720 | 66,061 | 44% | 5,162 | 

Parameters of corpus generation (columns 2 - 7 section) and collection results (columns 8 - 11). 2-4 excerpts were extracted from each text. Texts were split evenly among genres. Each excerpt is approximately 200 words long, extended to a sentence break. "Anchors" indicates how many times each excerpt is the anchor of a triplet. 7 judgments are collected per triplet, then cleaned to produce good-faith judgments. High agreement triplets are calculated as those in which 3 more judgments are one answer than the other; see paper section "Modeling Style with the Similarity Dataset" for more detail.
