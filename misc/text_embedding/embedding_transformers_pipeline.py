from transformers import pipeline

## no progress ba
# feature_extraction = pipeline('feature-extraction', model="emilyalsentzer/Bio_ClinicalBERT", tokenizer="emilyalsentzer/Bio_ClinicalBERT")

# features = feature_extraction(["Hello I'm a single sentence",
#                                "And another sentence",
#                                "And the very very last one"])

class ListDataset(Dataset):
    
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]
    

# clinical BERT text encoder
def clinical_text_encoder(dataset):
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", model_max_length=512, padding=True, truncation=True)
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    pipe = pipeline('feature-extraction', model=model, 
                tokenizer=tokenizer)
    
    # features = tqdm(pipe(text))
    # tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
    features = [i for i in tqdm(pipe(dataset, max_length=512, padding="max_length", truncation=True))]
    features = np.squeeze(features)
    return features