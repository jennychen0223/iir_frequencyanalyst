import streamlit as st
import xml.etree.ElementTree as ET
import re
import os 
import json

from src.upload_file import upload_file
from src.search_engine import search_engine
from src.download_pubmed import download_pubmed
from src.frequency_analyst import frequency_analyst
# st.set_page_config(page_title="Search Engine system")

from streamlit_option_menu import option_menu

import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
st.set_page_config(layout="wide")


with st.sidebar:
    selected = option_menu("Main Menu", ["Search Engine", "Download PubMed", "Frequency Analyst"],
                           icons=['search-heart-fill','cloud-arrow-down-fill', 'lightbulb-fill'], 
                           menu_icon="bars", default_index=0)
    
    # selected = option_menu("Main Menu", ["Upload File", "Search Engine", "Download PubMed", "Frequency Analyst"],
    #                        icons=['cloud-upload-fill', 'search-heart-fill','cloud-arrow-down-fill', 'lightbulb-fill'], 
    #                        menu_icon="bars", default_index=0)
    
if selected == "Search Engine":
    search_engine()

elif selected == "Upload File":
    upload_file()

elif selected == "Download PubMed":
    download_pubmed()

elif selected == "Frequency Analyst":
    frequency_analyst()

