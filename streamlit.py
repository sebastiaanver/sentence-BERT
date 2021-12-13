import streamlit as st
import pandas as pd
import numpy as np
from scipy import spatial

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

