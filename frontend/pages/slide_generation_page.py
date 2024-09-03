import streamlit as st

def main():
    st.title("Slide Generation")
    st.write("This is the slide generation page.")
    file_dir = st.text_input("Enter the directory path for slide generation:")
    if st.button("Generate Slides"):
        st.write(f"Slides will be generated for the directory: {file_dir}")
        # Here you would typically call the backend API to trigger slide generation
