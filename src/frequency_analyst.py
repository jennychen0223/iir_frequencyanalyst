import streamlit as st
from streamlit_option_menu import option_menu
import xml.etree.ElementTree as ET
from src.utils import parse_xml
from src.utils import search_and_highlight
from src.utils import *
import re
from Bio import Entrez
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import PorterStemmer

nltk.download('punkt')

def zip_distribution(documents, top_of_word, keyword_search):

    file_path = f"save_img/zip_{keyword_search}_{top_of_word}.png"
    filtered_tokens = []
    if os.path.exists(file_path):
        st.markdown(f'<p style="text-align:center; color:red;">Table: Top {top_of_word} Words | Zipf Distribution of Terms</p>', unsafe_allow_html=True)
        st.image(file_path)
    else: 
        progress_text = "Please wait! Processing ..."
        my_bar = st.progress(0, text=progress_text)
        process = 0
        for doc in documents:
            tokens = clean_and_tokenize(doc)
            filtered_tokens.extend(tokens)
            process += int(100/len(documents))
            my_bar.progress( process  , text=progress_text)
        my_bar.empty()

        # Calculate word frequencies
        word_freq = Counter(filtered_tokens)
        # 根據頻率排序
        sorted_word_counts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        # print(sorted_word_counts)
        
        # 提取詞和頻率
        ranks = range(1, len(sorted_word_counts) + 1)
        frequencies = [count for word, count in sorted_word_counts]

        # 使用Columns來佈局
        col1, col2 = st.columns([2, 1])

        with col1:

            # 畫Zipf分布
            plt.figure(figsize=(10, 6))
            plt.plot(ranks, frequencies, marker='o', linestyle='--', color='r')

            # # 添加文字標籤
            # for i, (word, count) in enumerate(sorted_word_counts[:top_of_word]):  # 只標記前8個詞
            #     plt.text(ranks[i], frequencies[i], f"{word}", fontsize=20, ha='right')

            # 添加文字標籤，避開點的位置
            for i, (word, count) in enumerate(sorted_word_counts[:top_of_word]):
                plt.annotate(f"{word}",
                            (ranks[i], frequencies[i]),
                            textcoords="offset points", xytext=(5, 5), ha='left')

            plt.title('Zipf Distribution')
            plt.xlabel('Rank Order of Frequency')
            plt.ylabel('Occurrence of Words')
            plt.grid(True)
            
            # Display the plot
            st.pyplot(plt)

        with col2:

            # Display DataFrame
            df = pd.DataFrame(sorted_word_counts, columns=['Word', 'Frequency'])
            df.index = df.index + 1
            st.dataframe(df.head(top_of_word).style.set_properties(**{'text-align': 'left'}),
                         use_container_width=True)
            
            # st.dataframe(df.head(top_of_word))  # 增加表格高度

def remove_stopwords(documents, top_of_word, keyword_search):
    filtered_tokens = []
    file_path = f"save_img/stopwords_{keyword_search}_{top_of_word}.png"
    if os.path.exists(file_path):
        st.markdown(f'<p style="text-align:center; color:red;">Table: Top {top_of_word} Words | Zipf Distribution of Terms (Remove Stopwords)</p>', unsafe_allow_html=True)
        st.image(file_path)
    else: 
        progress_text = "Please wait! Processing ..."
        my_bar = st.progress(0, text=progress_text)
        process = 0
        for doc in documents:
            tokens = clean_and_tokenize(doc)
            filtered_tokens.extend([word for word in tokens if word not in stopwords.words('english')])
            process += int(100/len(documents))
            my_bar.progress( process  , text=progress_text)
        my_bar.empty()

        # Calculate word frequencies
        word_freq = Counter(filtered_tokens)
        # 根據頻率排序
        sorted_word_counts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        # number_of_words = top_of_word
        # top_100_words = dict(list(sorted_word_counts.items())[:number_of_words])
        
        # 提取詞和頻率
        ranks = range(1, len(sorted_word_counts) + 1)
        frequencies = [count for word, count in sorted_word_counts]

        col1, col2 = st.columns([2, 1])
        with col1:
            # 畫Zipf分布
            plt.figure(figsize=(10, 6))
            plt.plot(ranks, frequencies, marker='o', linestyle='--', color='r')

            # 添加文字標籤，避開點的位置
            for i, (word, count) in enumerate(sorted_word_counts[:top_of_word]):
                plt.annotate(f"{word}",
                            (ranks[i], frequencies[i]),
                            textcoords="offset points", xytext=(5, 5), ha='left', fontsize=9)

            plt.title('Remove Stopwords')
            plt.xlabel('Rank Order of Frequency')
            plt.ylabel('Occurrence of Words')
            plt.grid(True)
            
            # Display the plot
            st.pyplot(plt)

        with col2:

            # Display DataFrame
            df = pd.DataFrame(sorted_word_counts, columns=['Word', 'Frequency'])
            df.index = df.index + 1
            st.dataframe(df.head(top_of_word).style.set_properties(**{'text-align': 'left'}),
                         use_container_width=True)


def porter_stemmer(documents, top_of_word, keyword_search, remove_stopwords = True): 
    # Apply Porter's stemming algorithm to the filtered tokens

    file_path = f"save_img/porter_{keyword_search}_{top_of_word}_{remove_stopwords}.png"
    if os.path.exists(file_path):
        st.markdown(f'<p style="text-align:center; color:red;">Table: Top {top_of_word} Words | Zipf Distribution of Terms (Stopwords Removed and Porter Stemming)</p>', unsafe_allow_html=True)
        st.image(file_path)
    else: 
        filtered_tokens = [] 
        if remove_stopwords:
            progress_text = "Please wait! Processing ..."
            my_bar = st.progress(0, text=progress_text)
            process = 0
            for doc in documents:
                tokens = clean_and_tokenize(doc)
                filtered_tokens.extend([word for word in tokens if word not in stopwords.words('english')])
                process += int(100/len(documents))
                my_bar.progress( process  , text=progress_text)
            my_bar.empty() 
        else:
            progress_text = "Please wait! Processing ..."
            my_bar = st.progress(0, text=progress_text)
            process = 0
            for doc in documents:
                tokens = clean_and_tokenize(doc)
                filtered_tokens.extend(tokens)
                process += int(100/len(documents))
                my_bar.progress( process  , text=progress_text)
            my_bar.empty() 
            
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

        # Calculate word frequencies
        word_freq = Counter(stemmed_tokens)
        # 根據頻率排序
        sorted_word_counts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        # 提取詞和頻率
        ranks = range(1, len(sorted_word_counts) + 1)
        frequencies = [count for word, count in sorted_word_counts]

        col1, col2 = st.columns([2, 1])
        with col1:
            # 畫Zipf分布
            plt.figure(figsize=(10, 6))
            plt.plot(ranks, frequencies, marker='o', linestyle='--', color='r')

            # # 添加文字標籤
            # for i, (word, count) in enumerate(sorted_word_counts[:8]):  # 只標記前8個詞
            #     plt.text(ranks[i], frequencies[i], f"{word}", fontsize=9, ha='right')


            # 添加文字標籤，避開點的位置
            for i, (word, count) in enumerate(sorted_word_counts[:top_of_word]):
                plt.annotate(f"{word}",
                            (ranks[i], frequencies[i]),
                            textcoords="offset points", xytext=(5, 5), ha='left', fontsize=9)


            plt.title('Porter’s algorithm')
            plt.xlabel('Rank Order of Frequency')
            plt.ylabel('Occurrence of Words')
            plt.grid(True)
            
            # Display the plot
            st.pyplot(plt)

        with col2:

            # Display DataFrame
            df = pd.DataFrame(sorted_word_counts, columns=['Word', 'Frequency'])
            df.index = df.index + 1
            st.dataframe(df.head(top_of_word).style.set_properties(**{'text-align': 'left'}),
                         use_container_width=True)
            


def compare(documents,top_of_word, keyword_search, remove_stopwords = True):
    file_path = f"save_img/compare_{keyword_search}_{top_of_word}_{remove_stopwords}.png"
    if os.path.exists(file_path):
        st.markdown(f'<p style="text-align:center; color:red;">Table: Comparison Before and After Applying Porter Algorithm </p>', unsafe_allow_html=True)
        st.image(file_path)
    else: 
        
        filtered_tokens = [] 
        if remove_stopwords:
            progress_text = "Please wait! Processing ..."
            my_bar = st.progress(0, text=progress_text)
            process = 0
            for doc in documents:
                tokens = clean_and_tokenize(doc)
                filtered_tokens.extend([word for word in tokens if word not in stopwords.words('english')])
                process += int(100/len(documents))
                my_bar.progress( process  , text=progress_text)
            my_bar.empty() 
        else:
            progress_text = "Please wait! Processing ..."
            my_bar = st.progress(0, text=progress_text)
            process = 0
            for doc in documents:
                tokens = clean_and_tokenize(doc)
                filtered_tokens.extend(tokens)
                process += int(100/len(documents))
                my_bar.progress( process  , text=progress_text)
            my_bar.empty() 
            
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

        # Calculate word frequencies before stemming
        word_freq_before = Counter(filtered_tokens)

        # Calculate word frequencies after stemming
        word_freq_after = Counter(stemmed_tokens)

        # 根據頻率排序
        sorted_word_freq_before = sorted(word_freq_before.items(), key=lambda x: x[1], reverse=True)
        sorted_word_freq_after = sorted(word_freq_after.items(), key=lambda x: x[1], reverse=True)

        # # Display only the top 20 frequencies
        # number_of_words = top_of_word
        # top_words_before = dict(list(sorted_word_freq_before.items())[:number_of_words])
        # top_words_after = dict(list(sorted_word_freq_after.items())[:number_of_words])

        # 提取詞和頻率
        ranks1 = range(1, len(sorted_word_freq_before) + 1)
        frequencies1 = [count for word, count in sorted_word_freq_before]

        ranks2 = range(1, len(sorted_word_freq_after) + 1)
        frequencies2 = [count for word, count in sorted_word_freq_after]
        
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            # 畫Zipf分布
            plt.figure(figsize=(10, 6))
            plt.plot(ranks1, frequencies1, marker='o', linestyle='--', color='r', label='Without Porter’s algorithm')
            plt.plot(ranks2, frequencies2, marker='o', linestyle='--', color='b', label='Porter’s algorithm')

            # # 添加文字標籤，只標記前3個詞
            # for i, (word, count) in enumerate(sorted_word_freq_before[:3]):
            #     plt.text(ranks1[i], frequencies1[i], f"{word}", fontsize=9, ha='right', color='r')
            # for i, (word, count) in enumerate(sorted_word_freq_after[:3]):
            #     plt.text(ranks2[i], frequencies2[i], f"{word}", fontsize=9, ha='right', color='b')


            # 添加文字標籤，避開點的位置
            for i, (word, count) in enumerate(sorted_word_freq_before[:top_of_word]):
                plt.annotate(f"{word}",
                            (ranks1[i], frequencies1[i]),
                            textcoords="offset points", xytext=(5, -15), ha='left', va='top', fontsize=9)

            # 添加文字標籤，避開點的位置
            for i, (word, count) in enumerate(sorted_word_freq_after[:top_of_word]):
                plt.annotate(f"{word}",
                            (ranks2[i], frequencies2[i]),
                            textcoords="offset points", xytext=(5, 5), ha='left', va='bottom', fontsize=9)


            plt.title('Compare')
            plt.xlabel('Rank Order of Frequency')
            plt.ylabel('Occurrence of Words')
            plt.grid(True)
            plt.legend()
            
            st.pyplot(plt)

        with col2:
            st.subheader("Without Porter’s algorithm")

            # Display DataFrame
            df1 = pd.DataFrame(sorted_word_freq_before, columns=['Word', 'Frequency'])
            df1.index = df1.index + 1
            st.dataframe(df1.head(top_of_word).style.set_properties(**{'text-align': 'left'}),
                         use_container_width=True)

        with col3:
            st.subheader("Porter’s algorithm")

            # Display DataFrame
            df2 = pd.DataFrame(sorted_word_freq_after, columns=['Word', 'Frequency'])
            df2.index = df2.index + 1
            st.dataframe(df2.head(top_of_word).style.set_properties(**{'text-align': 'left'}),
                         use_container_width=True)
            
    
def frequency_analyst(): 
    keyword_search = ''
    keyword_search = st.text_input("Enter keyword :")
    edit_distance = st.toggle("Edit distance", value=False)
    path_keywords = os.listdir("dataset")

    top_of_word = st.number_input("Top of words", min_value=5, step=1, format="%d")

    if st.button("Start Analyst"):
        
        if keyword_search in path_keywords: 
            st.info(f"Your keyword is {keyword_search}", icon="ℹ️")

        elif edit_distance and len(keyword_search) > 0: 

            suggestions = find_closest_keywords(keyword_search, path_keywords, num_suggestions = 10)
            keywords, probabilities = zip(*suggestions)
            data = {"Keywords": keywords, "Probability": probabilities}
            df = pd.DataFrame(data)
        
            if probabilities[0] > 0.6: 
                st.info(f"Closest keywords to '{keyword_search}': {keywords[0]}", icon="ℹ️")
                keyword_search = keywords[0]
            else:
                st.warning("No found the keyword", icon="⚠️")
                keyword_search = '' 

            col1, col2 = st.columns([2, 1])

            with col1:
                # Create a bar chart using Seaborn               

                plt.figure(figsize=(8, 4))
                sns.set(style="whitegrid")  # Set the style to have a white grid
                ax = sns.barplot(x="Keywords", y="Probability", data=df)
                plt.xticks(rotation=45)
                plt.title("Keyword Probabilities with Edit Distance")

                # Add percentages on each bar
                for p in ax.patches:
                    ax.annotate(f'{p.get_height()*100:.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                textcoords='offset points')

                st.pyplot(plt)
            with col2:
                st.dataframe(df.head(top_of_word).style.set_properties(**{'text-align': 'left'}),
                            use_container_width=True)

        elif len(keyword_search) > 0: 
            st.warning("No found the keyword", icon="⚠️")
            keyword_search = ''

        if len(keyword_search) > 0: 
            # st.sidebar.title("Setting")
            documents = []
            list_file = os.listdir(f"dataset/{keyword_search}")
            for file in list_file: 
                documents.append(parse_xml_to_string(f"dataset/{keyword_search}/{file}"))

            zip_distribution(documents, top_of_word, keyword_search)
            st.write("---")
            remove_stopwords(documents, top_of_word, keyword_search)
            st.write("---")
            porter_stemmer(documents, top_of_word, keyword_search)
            st.write("---")
            compare(documents, top_of_word, keyword_search)
