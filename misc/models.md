## Clinical BERT

Defualt baseline model

## RoBERTa-large-PM-M3-Voc

(referred to as BioClinRoBERTa in "Do we still need Clinial LMs?")

- a RoBERTa model pre-trained on PubMed and PMC and MIMIC-III with a BPE Vocab learnt from PubMed

- best model in [Pretrained Language Models for Biomedical and Clinical Tasks: Understanding and Extending the State-of-the-Art](https://aclanthology.org/2020.clinicalnlp-1.17/)

- best model in "Do we still need Clinial LMs?"

- [link](https://github.com/facebookresearch/bio-lm)

## Clinical Longformer

- [Revisiting Transformer-based Models for Long Document Classification](https://arxiv.org/pdf/2204.06683.pdf)

- `Max sequence	= 4096`

- [model card](https://huggingface.co/xdai/mimic_longformer_base)

- [link](https://github.com/coastalcph/trldc/tree/main)

- hierarchical transformers for long document classification

- data: discharge summaries from MIMIC-III