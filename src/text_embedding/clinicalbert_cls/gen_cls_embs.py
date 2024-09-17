import pickle
from simpletransformers.language_representation import RepresentationModel
from simpletransformers.config.model_args import ModelArgs

# batch is an iterable, list/Series/
batch = ['clinical note', 'another clinical note']

model_args = ModelArgs(max_seq_length=512, silent = True)
model = RepresentationModel(
    "bert", "emilyalsentzer/Bio_ClinicalBERT", args=model_args)


# features = model.encode_sentences(batch, combine_strategy="mean")
# features = model.encode_sentences(batch, combine_strategy=None)
# CLS
features = model.encode_sentences(batch, combine_strategy=0)