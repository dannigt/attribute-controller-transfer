# Data and Preprocessing

## Data

### Formality Control Data

#### Training
The training data come from [CoCoA-MT](https://github.com/amazon-science/contrastive-controlled-mt/tree/main/CoCoA-MT).

At the time of writing, there are six languages in this dataset: en-{de, es, fr, it, hi, ja}.

We excluded ja, since NLLB has very low translation accuracy on formality-annotated words (<40%, whereas all 5 other languages score >60%).

#### Evaluation

##### Source Transfer
We re-align the CoCoA-MT test set using English as pivot, leading to non-English source and target sentences. 

We share our re-aligned data in `./data/formality/source_transfer`.

##### Target Transfer
The test data are from the [IWSLT 2023 formality control task](https://github.com/amazon-science/contrastive-controlled-mt/tree/main/IWSLT2023).

### Gender Control Data

#### Training
The training data come from the en-es [adaptation set](https://github.com/DCSaunders/tagged-gender-coref#adaptation-sets) by Saunders et al. (2020). 

#### Evaluation
For evaluatation we use the [MuST-SHE](https://aclanthology.org/2020.acl-main.619.pdf) test set.

The dataset provides en-{es, it, fr} data as well multiway aligned data. 

## Preprocessing

### Training
When training the classifier in classifier guidance,
the target attribute is passed to the model by a token in the source sentence (following the src token).

Our implementation ignores this token in the input to the encoder
 (`L62-63` of `fairseq-nllb/examples/classifier_guidance/classifier_guidance_src/criterions/attribute_classification_cross_entropy.py`).   

### Decoding
For CoCoA-MT, 
many test inputs contain multiple sentences. 
When directly decoding, NLLB-200 suffered from severe under-translation,
where the output translation only contains one sentence.

We therefore split the input by sentence boundaries and decode sentence by sentence.