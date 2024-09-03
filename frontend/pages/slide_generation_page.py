import streamlit as st
import requests
import sseclient



def main():
    st.title("Slide Generation")

    # Sidebar with form
    with st.sidebar:
        with st.form(key='slide_gen_form'):
            file_dir = st.text_input("Enter the directory path for slide generation:",
                                     value="./data/summaries_test",
                                     placeholder="./data/summaries_test")
            submit_button = st.form_submit_button(label='Submit')

    # Main view divided into two columns
    left_column, right_column = st.columns(2)

    with left_column:
        with st.expander("Workflow execution Status"):
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
            # Send a request to the backend
            with st.empty():
                response = requests.post("http://backend:80/run-slide-gen", json={"path": file_dir})
                client = sseclient.SSEClient(response)

                full_response = ""
                for event in client.events():
                    full_response += event.data
                    st.write(full_response)

                # if response.status_code == 200:
                #     dynamic_content.markdown(f"Slides will be generated for the directory: {file_dir}")
                # else:
                #     dynamic_content.markdown("Failed to send request to backend.")


main()
