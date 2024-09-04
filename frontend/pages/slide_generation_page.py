import threading

import streamlit as st
import requests
import sseclient
import httpx
import asyncio


async def fetch_streaming_data(url: str, payload: dict = None):
    async with httpx.AsyncClient(timeout=1200.0) as client:
        # Making an async request to the FastAPI stream endpoint
        async with client.stream("POST", url=url, json=payload) as response:
            # Iterating through the streamed lines
            async for line in response.aiter_lines():
                if line:
                    print(f"Received line: {line}")  # Debugging print
                    yield line


# Function to display the streaming data in the Streamlit UI
async def get_stream_data(url, payload, result_placeholder=None):
    # async for line in fetch_streaming_data(url, payload):
    #     print(f"Displaying line: {line}")  # Debugging print
    #     result_placeholder.write(line)  # Updating the UI with each streamed line

    async for line in fetch_streaming_data(url, payload):
        st.write(line)  # Directly write the line to the UI


# def display_streaming_data(url, payload):
#     # Streamlit's empty placeholder to display the streaming data
#     result_placeholder = st.empty()
#
#     # Create an asyncio task to fetch and display streaming data in the background
#     asyncio.create_task(get_stream_data(url, payload, result_placeholder))
# Function to run the async function in a thread
def run_streaming_in_thread(url, payload):
    result_placeholder = st.empty()  # Placeholder to display results

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(get_stream_data(url, payload, result_placeholder))


def display_streaming_data(url, payload):
    # Start the async streaming in a separate thread
    thread = threading.Thread(target=run_streaming_in_thread, args=(url, payload))
    thread.start()


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
        # dynamic_content = st.empty()
        if submit_button:
            asyncio.run(get_stream_data("http://backend:80/run-slide-gen", {"path": "./data/summaries_test"}))
            # display_streaming_data("http://backend:80/run-slide-gen",
            #                        {"path": file_dir})

            # Send a request to the backend
            # try:
            #     response = requests.post("http://backend:80/run-slide-gen", json={"path": file_dir}, stream=True)
            #     # Pass the response object directly to SSEClient
            #     client = sseclient.SSEClient(response)
            #
            #     for event in client.events():
            #         if event.data:
            #             # Update the dynamic content
            #             dynamic_content.write(event.data)
            # except requests.exceptions.InvalidURL as e:
            #     st.error(f"Invalid URL error: {e}")
            # except Exception as e:
            #     st.error(f"An unexpected error occurred: {e}")
            #
            # # if response.status_code == 200:
            # #     dynamic_content.markdown(f"Slides will be generated for the directory: {file_dir}")
            # # else:
            # #     dynamic_content.markdown("Failed to send request to backend.")


main()
