## Experiments

Experiment documentations


| Name |  Architecture  | Data | Text Embedding | Runtime etc. | Additional Notes | 
| :---:   | :---: | :---: | :---: | :---: | :---: |
| `Exp00`| 0.0 | - | - |  |  |
| `Exp0` | 0 | `sepsis_removed_0.pkl` | __Clinical BERT__ | - | - |
| `Exp1` | 0 | `sepsis_removed_1.pkl` | __Clinical BERT__ | - | - |
| `Exp2` | 0 | `sepsis_removed_1.pkl` | __RoBERTa-large-PM-M3-Voc__ | - | - |
| `Exp3` | 0 | `sepsis_removed_1.pkl` | __Clinical Longformer__ | - | - |
| `Exp4` | 0 | `sepsis_removed_2.pkl` | __GloVe__ | - | - |


### Variations in each experiment:

- training epochs

- architecture-wise: layers following text embedding module
    - +/- `Dropout` layers
    - +/- dense layers, varied activation functions

- sentence embedding:
    - average
    - concat
    - \[CLS\]
    - [bert-as-a-service?](https://github.com/jina-ai/clip-as-service)



#### variations general-architecture-

- Architecture 0.0: text embeddings compressed into a number 

- Architecture 0: org in research module

- Architecture 1: integrate time and variable name in the text embedding module

- Architecture 2: ?