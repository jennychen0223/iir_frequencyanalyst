import os
import streamlit as st 
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn


def cal(words1, words2):    

    cbow_model = Word2Vec.load("cbow_model.model")
    skipgram_model = Word2Vec.load("skipgram_model.model")
    
    word_vectors_cbow = cbow_model.wv
    similarity_cbow = word_vectors_cbow.similarity(words1, words2)

    word_vectors_skipgram= skipgram_model.wv
    similarity_skip = word_vectors_skipgram.similarity(words1, words2)

    similarity_cbow = round(similarity_cbow, 2)
    similarity_skip = round(similarity_skip, 2)

    return similarity_cbow, similarity_skip


# nltk.download('punkt') 
def bag_of_word():
    
    st.markdown(f'<p style="text-align:center; color:red;">Wordcloud</p>', unsafe_allow_html=True)
    st.image("image/wordcloud.jpg")
    # st.markdown(f'<p style="text-align:center; color:red;">Training Loss</p>', unsafe_allow_html=True)
    st.image("image/未命名.jpg")

    # Search
    keyword1 = st.text_input("Input context word 1 :")
    keyword2 = st.text_input("Input context word 2 :") 
    # keyword3 = st.text_input("Input context word W(t+1) :")
    # keyword4 = st.text_input("Input context word W(t+2) :")
    if len(keyword1) > 0 and len(keyword2) > 0:
        st.info(f"Your context of word W1 = {keyword1}, W2 = {keyword2}", icon="ℹ️")

    run_cbow = st.button("Calculate Similarity")
    if run_cbow:
        similarity_cbow, similarity_skip = cal(keyword1, keyword2)

        st.write(f"Similarity between {keyword1} and {keyword2}: {similarity_cbow} with CBOW")
        st.write(f"Similarity between {keyword1} and {keyword2}: {similarity_skip} with Skip-Gram")
