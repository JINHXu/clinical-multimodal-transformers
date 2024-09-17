# https://www.scaler.com/topics/nlp/huggingface-transformers/

from transformers import BertTokenizer, TFBertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertModel.from_pretrained("bert-base-cased")
text_1 = "Replace me by any text you'd like."
encoded_text = tokenizer(text_1, return_tensors='tf')
output = model(encoded_text)




import tensorflow as tf

class AttentionBottleneck(tf.keras.Model):
    def __init__(self, input_dim_modal1, input_dim_modal2, hidden_dim, num_classes):
        super(AttentionBottleneck, self).__init__()

        # Define the attention mechanism
        self.attention = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='tanh'),
            tf.keras.layers.Dense(1, activation='softmax')
        ])

        # Define fusion layers if needed
        self.fc = tf.keras.layers.Dense(hidden_dim)

        # Define task-specific layers
        self.task_specific_layer = tf.keras.layers.Dense(num_classes)

    def call(self, modal1_features, modal2_features):
        # Compute attention weights
        combined_features = tf.concat([modal1_features, modal2_features], axis=1)
        attention_weights = self.attention(combined_features)

        # Weighted fusion
        fused_features = attention_weights[:, 0] * modal1_features + attention_weights[:, 1] * modal2_features

        # Apply fusion layers if needed
        fused_features = self.fc(fused_features)

        # Task-specific processing
        output = self.task_specific_layer(fused_features)
        return output


