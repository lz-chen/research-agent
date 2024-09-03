import streamlit as st

import streamlit as st
from pages.main_page import main as main_page
from pages.slide_generation_page import main as slide_generation_page

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main Page", "Slide Generation"])

if page == "Main Page":
    main_page()
elif page == "Slide Generation":
    slide_generation_page()
