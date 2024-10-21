import streamlit as st
from streamlit_option_menu import option_menu
import xml.etree.ElementTree as ET
import xml.etree.ElementTree as ET
from src.utils import parse_xml, parse_xml_to_string
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
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def search_engine():
    # Search
    keyword = st.text_input("Search Engine (Enter the keyword) : ")
    case_sensitive = True
    # case_sensitive = st.toggle("Case Sensitive Search", value=True)
    matching_articles = []
    # edit_distance = st.toggle("Edit distance", value=False)
    edit_distance = True

    if edit_distance and len(keyword) > 0: 
        data = []
        documents = []
        # Load uploaded files
        filtered_tokens = [] 

        folders = os.listdir("dataset")
        keyword = 'enterovirus'

        for folder in folders:
            if folder == keyword or folder == 'upload':

                list_file = os.listdir(os.path.join("dataset", folder))

                if len(list_file) != 0:

                    for file in list_file:
                        try:
                            data += parse_xml(os.path.join("dataset", folder, file))
                            documents.append(parse_xml_to_string(os.path.join("dataset", folder, file)))
                        except: 
                            continue 

        for doc in documents:
          tokens = clean_and_tokenize(doc)
          filtered_tokens.extend([word for word in tokens if word not in stopwords.words('english')])

        len(filtered_tokens)
        unique_list = list(set(map(str.lower, filtered_tokens)))

        filtered_list = [word for word in unique_list if not word.isnumeric() and word.isalpha() and len(word) <= 10]

        # Calculate similarity scores for all keywords
        similarity_scores = [(word, fuzz.ratio(keyword, word)) for word in filtered_list]
        # Sort the list of keywords by similarity in descending order
        sorted_keywords = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        # Get the top 10 keywords
        top_10_keywords = [word for word, score in sorted_keywords[:9]]
        top_10_keywords.append(keyword)

        # option = st.selectbox('Recommend the nearest word...',top_10_keywords)

        # st.info(f"Your keyword: {option}", icon="ℹ️")
        # keyword = option        

        # print("Top 10 keywords:")
        # for keyword in top_10_keywords:
        #     print(keyword)
          

    if st.button("Search"):
        for article in data:
            highlighted_fields = search_and_highlight(article, keyword, case_sensitive)
            if any(isinstance(value, str) and '<span style="background-color: yellow">' in value for value in highlighted_fields.values()):
                matching_articles.append(highlighted_fields)

        if not matching_articles:
            st.error("Keywords not found.")
        else:
            st.success("Successfully found Keywords")

            for idx, article in enumerate(matching_articles, start=1):
                file_name = article['PMID']
                st.markdown(f'<p style="text-align:center; color:red;">Matching Article {idx}: {file_name}.xml</p>', unsafe_allow_html=True)
          
                # Calculate and display line count for abstract
                abstract_text = article['Abstract']

                num_lines = len(abstract_text.split('\n')) if abstract_text else 0
                # st.markdown(f"**Number of Lines in Abstract**: {num_lines}", unsafe_allow_html=True)
                
                ## Keywords
                num_keywords = len(keyword.split())

                ## Characters
                try:              
                    # 包含空格的字符數
                    num_characters_including_spaces = len(abstract_text)
                    # 不包含空格的字符數
                    num_characters_excluding_spaces = len(abstract_text.replace(" ", ""))
                    
                except TypeError:
                    num_characters_including_spaces = 0
                    num_characters_excluding_spaces = 0

                ## Words
                try:              
                    num_words = len(abstract_text.split())
                except (TypeError, AttributeError):
                    num_words = 0
                
                ## Sentences          
                sss = re.split(r'[.!?]', abstract_text)
                num_sentences = len([s for s in sss if s.strip()])

                ## non-ASCII
                try:
            
                    # 非 ASCII 字符數
                    num_non_ascii_characters = sum(1 for char in abstract_text if ord(char) > 127)

                    # 非 ASCII 單詞數
                    words = re.findall(r'\b\w+\b', abstract_text)
                    num_non_ascii_words = sum(1 for word in words if any(ord(char) > 127 for char in word))
                except TypeError:
            
                    num_non_ascii_characters = 0
                    num_non_ascii_words = 0
              

                # Create a table for document statistics
                statistics_table = {
                    "Statistic": ["Number of Keywords", 
                                    "Number of Characters (including spaces)", 
                                    "Number of Characters (excluding spaces)",  
                                    "Number of Words",
                                    "Number of Sentences",
                                    "Number of non-ASCII characters",
                                    "Number of non-ASCII words",                            
                                    ],

                    "Value": [num_keywords, 
                                num_characters_including_spaces, 
                                num_characters_excluding_spaces, 
                                num_words,
                                num_sentences,
                                num_non_ascii_characters,
                                num_non_ascii_words
                                ]
                }

                st.table(statistics_table)
                        

                # Display other article information
                for key, value in article.items():
                    if key in ['PMID', 'Title', 'Journal Title', 'ISSN', 'Publication Date', 'Authors', 'Keywords']:
                        # Format these fields as bold and italic
                        st.markdown(f"**_{key}_**: {value}", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{key}**: {value}", unsafe_allow_html=True)
                st.write("---")

