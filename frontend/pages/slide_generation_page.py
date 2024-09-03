import streamlit as st


def main():
    st.title("Slide Generation")

    # Sidebar with form
    with st.sidebar:
        with st.form(key='slide_gen_form'):
            file_dir = st.text_input("Enter the directory path for slide generation:")
            submit_button = st.form_submit_button(label='Submit')

    # Main view divided into two columns
    left_column, right_column = st.columns(2)

    with left_column:
        with st.expander("Options"):
            st.write("Additional options can be placed here.")
        
        st.write("Chat View")
        # Placeholder for chat messages
        chat_messages = st.empty()

    with right_column:
        # Placeholder for dynamic content
        dynamic_content = st.empty()
        if submit_button:
            dynamic_content.markdown(f"Slides will be generated for the directory: {file_dir}")
            # Here you would typically call the backend API to trigger slide generation


main()
