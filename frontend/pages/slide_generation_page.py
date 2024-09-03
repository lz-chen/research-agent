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

        # Chat input and message display
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display existing chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input box
        if user_input := st.chat_input("Type your message"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(user_input)

            # Here you can add logic to generate a response from a model or API
            response = "This is a placeholder response."

            # Add response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)

    with right_column:
        # Placeholder for dynamic content
        dynamic_content = st.empty()
        if submit_button:
            dynamic_content.markdown(f"Slides will be generated for the directory: {file_dir}")
            # Here you would typically call the backend API to trigger slide generation


main()
