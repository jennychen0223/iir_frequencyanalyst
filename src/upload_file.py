import streamlit as st
import xml.etree.ElementTree as ET

def upload_file():
    # Sidebar
    # st.sidebar.title("Documents Area")
    # uploaded_files_xml = st.sidebar.file_uploader("Upload Files (XML)", type=["xml"], accept_multiple_files=True)

    st.title("Upload XML File")

    # 使用st.file_uploader來創建一個檔案上傳區塊
    uploaded_file = st.file_uploader("選擇檔案或拖曳至此", type=["xml"], accept_multiple_files=True)

    # Initialize data list
    data = []

    # Load uploaded files
    for xml_file in uploaded_file:
        print(xml_file)
        
    #     st.success(' Download success!', icon="✅")