# MA-Thesis

Ongoing experiments: documentations, scripts and etc.

### Problems with data handling in research module:

- involuntary truncation at 512
- text preprocessing


## Architecture 0.0

each text vector down to one number?

## Architecture 0 (same architecture in Research Module)

### Clinical BERT

#### Data Preprocessing

Clinical notes preprocessing in research module: stop words and special chars removal, case normalisation following wang et al.

-> this was not right!

- clinical BERT (a contextual language model), stop words were not supposed to be removed
- case normalisation was also tricky for clinical notes preprocessing, e.g. the medical condition ADD (attention deficit disorder) may be converted to the verb add by pre‐processing, if a blanket standardization to lowercase is applied (according to a survey paper)
- special chars removal? same special chars were input to the pretraining procesure of Clinical BERT



### GloVe

- data cleaning 
- 

### Another text embedding module


## Architecture 1

text emebedding + time information ?


...

## Another dataset!
