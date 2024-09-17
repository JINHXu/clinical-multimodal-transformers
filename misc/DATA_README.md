## Data

### Datasets created in the process

| Name |  Sepsis keywords  | Text Preprocessing | additional notes | target model | runtime |
| :---:   | :---: | :---: | :---: | :---: | :---: |
| `research_module_data.pkl` | `sepsis` and `septic` | full-house data cleaning in research module following wang et al. | `nltk` for sentence extraction | ClinicalBERT | undocumented |
| `sepsis_removed_0.pkl` | `sepsis` and `septic` | well-reserved notes for CinicalBERT | `scispacy` for sentence extraction | ClinicalBERT | `1407430/1407430 [20:50:59<00:00, 18.75it/s]` |
| `sepsis_removed_1.pkl` | `sepsis` and `SIRS` | well-reserved notes for language models such as CinicalBERT | `scispacy` for sentence extraction | ClinicalBERT | undocumented |
| `sepsis_removed_2.pkl` | `sepsis` and `SIRS` | stop words removal, special chars removal, punctuations removal | `scispacy` for sentence extraction | GloVe | `1407430/1407430 [40:28:48<00:00,  9.66it/s]` |
| `sepsis_removed_3.pkl` | reserved | well-reserved notes for language models such as CinicalBERT | `scispacy` for sentence extraction | ClinicalBERT | undocumented |
| `sepsis_removed_4.pkl` | reserved | stop words removal, special chars removal, punctuations removal | `scispacy` for sentence extraction | GloVe |  |






### Text preprocessing

#### For LMs

wang et al. ❌

#### For GloVe

[text clean example](https://colab.research.google.com/drive/1gB1J5waZghzoeGbW5yyrloc7Ff6QrmHu?usp=sharing)

references:

- https://www.researchgate.net/post/GloVe_how_to_deal_with_tokens_which_have_punctuation_signs_inside_of_them_and_with_common_expression
- https://aclanthology.org/2020.figlang-1.7.pdf
