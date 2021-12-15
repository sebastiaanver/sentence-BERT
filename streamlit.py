import streamlit as st
import pandas as pd
import numpy as np
from scipy import spatial
from PIL import Image

from transformers import BertModel, BertTokenizer
from model import SentenceBertInference


# Read numpy files for search functionality
sentences = np.load("data/sentences.npy", allow_pickle=True)
vectors = np.load("data/vectors.npy", allow_pickle=True)
tree = spatial.KDTree(data=vectors)

# Init model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('sebastiaan/sentence-BERT-regression')
model = SentenceBertInference(tokenizer, bert_model)

# Results from experiments
experiments = ["Classification task", "Regression task", "Combined task"]
correlation = [50, 60, 70]
d = {'Experiment': experiments, 'Pearson correlation': correlation}
df = pd.DataFrame(data=d)
df.set_index('Experiment', inplace=True)

st.title('Sentence BERT')
st.title('Introduction')
st.write("The goal of this assignment is to train a network to produce sentence embedding, by implementing the proposed method of the Sentence-BERT (S-BERT) paper.")

st.title("Semantic Sentence Embedding")
st.write("As with most embedding system, the idea is to represent objects as a high-dimensional vectors, so that the "
         "distance correlates to a property in which we’re interested. In this case, we’re going to represent sentences "
         "so that the distance of the vector space indicates semantic similarity. Hence, sentences who have a similar "
         "meaning should end up close to each other, and sentences who are dissimilar in their meaning should be further apart")
st.code(
"""
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, col_names, tokenizer, device):
        self.sent_a_tensor = tokenizer(
            list(df[col_names[0]]),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        self.sent_b_tensor = tokenizer(
            list(df[col_names[1]]),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        self.labels = df[col_names[2]].values
        self.device = device
        
""", language = 'python')

st.title("Regression Training Objective")
st.write("The regression loss functions, takes into account the cosine-similarity between two sentence embedding. "
         "Hence, allowing us to directly tune the distance of the generated sentence embedding. "
         "Since cosine-similarity is always in the range [−1, 1], we can set the most similar sentences to have a "
         "similarity of 1, and the most dissimilar to have −1.")


image = Image.open('images/regression.png')
st.image(image, caption='SBERT architecture at inference, for example, to compute similarity scores. '
                        'This architecture is also used with the regression objective function', width=400)

st.write("We first need to scale the labels")
st.code("""train_df["scaled_score"] = train_df["score"].apply(lambda x: (float(x) / 2.5) - 1)
test_df["scaled_score"] = test_df["score"].apply(lambda x: (float(x) / 2.5) - 1)""", language='python')

st.write("If we implement the architectute, we can build SentenceBert")
st.code(
    """
    class SentenceBert(torch.nn.Module):
    def __init__(self, objective="cosine_similarity", bert_model=None):
        super(SentenceBert, self).__init__()
        config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bert_layer = bert_model if bert_model else BertModel.from_pretrained('bert-base-uncased', config=config)
        self.objective = objective
        if self.objective == "cosine_similarity":
            self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        elif self.objective == "classification":
            self.weights = torch.nn.Linear(in_features=768 * 3, out_features=3)
            self.softmax = torch.nn.Softmax()
        self.double()

    def forward(self, sent_a, sent_b):
        sent_a_out = self.bert_layer(**sent_a)
        sent_a_pooled = torch.mean(sent_a_out.last_hidden_state, dim=1)
        sent_b_out = self.bert_layer(**sent_b)
        sent_b_pooled = torch.mean(sent_b_out.last_hidden_state, dim=1)
        
        if self.objective == "cosine_similarity":
            return self.cos_sim(sent_a_pooled, sent_b_pooled)
        elif self.objective == "classification":
            subs = torch.abs(torch.sub(sent_a_pooled, sent_b_pooled))
            concat = torch.cat((sent_a_pooled, sent_b_pooled, subs), dim=1)
            logits = self.weights(concat)
            return self.softmax(logits)
        else:
            print("Objective not valid")
        return None
    
    """, language='python'


)
st.write("Evaluation on STS Benchmark")
st.write("The metric used to compare two sets of rankings is the Spearmean Correlation")

st.title('Experiments')
st.write("We have trained sentence-BERT by fine-tuning the pre-trained BERT model an various tasks and compared performance below. For the combined task we first fine-tuned on the classifcation task before fine-tuning on the regression task.")
st.table(df)

st.title('Search engine')
model_to_use = st.selectbox('Select model to use:', experiments)
query = st.text_input('Search query')

query_vec = model.predict(query).cpu().detach().numpy()
res = tree.query(query_vec, k=5)

st.text(f"Results for model: '{model_to_use}' and query: '{query}'")

for i in res[1]:
    st.text(sentences[i])

