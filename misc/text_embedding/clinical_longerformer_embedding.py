# https://huggingface.co/yikuan8/Clinical-Longformer

# ref

from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-Longformer")

output = model(**encoded_input, output_hidden_states=True)

print(outputs.keys())            
#odict_keys(['last_hidden_state', 'pooler_output', 'hidden_states'])

print("outputs[0] gives us sequence_output: \n", outputs[0].shape) #torch.Size([1, 34, 768])

print("outputs[1] gives us pooled_output  \n",outputs[1]) #Embeddings ( last hidden state) #[768]
            
print("outputs[2]: gives us Hidden_output: \n ",outputs[2][0].shape) #torch.Size([1, 512, 768]) 
     
