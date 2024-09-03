import streamlit as st

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main Page", "Slide Generation"])

if page == "Main Page":
    st.title("Main Page")
    st.write("Welcome to the main page of the Streamlit app.")
elif page == "Slide Generation":
    st.title("Slide Generation")
    st.write("This is the slide generation page.")
    file_dir = st.text_input("Enter the directory path for slide generation:")
    if st.button("Generate Slides"):
        st.write(f"Slides will be generated for the directory: {file_dir}")
        # Here you would typically call the backend API to trigger slide generation
