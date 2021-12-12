import streamlit as st
import pandas as pd

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
st.text(f"Results for model: '{model_to_use}' and query: '{query}'")

for i in range(5):
    st.text("This is a sentence result.")



