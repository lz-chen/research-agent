import json
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import httpx
import asyncio
import threading
import queue
import requests

# Initialize session state variables
if 'received_lines' not in st.session_state:
    st.session_state.received_lines = []

if 'workflow_id' not in st.session_state:
    st.session_state.workflow_id = None

if 'workflow_thread' not in st.session_state:
    st.session_state.workflow_thread = None

if 'user_input_required' not in st.session_state:
    st.session_state.user_input_required = False

if 'user_input_prompt' not in st.session_state:
    st.session_state.user_input_prompt = None

if 'message_queue' not in st.session_state:
    st.session_state.message_queue = queue.Queue()

if 'user_input_event' not in st.session_state:
    st.session_state.user_input_event = threading.Event()


async def fetch_streaming_data(url: str, payload: dict = None):
    async with httpx.AsyncClient(timeout=1200.0) as client:
        async with client.stream("POST", url=url, json=payload) as response:
            async for line in response.aiter_lines():
                if line:
                    yield line


async def get_stream_data(url, payload, message_queue, user_input_event):
    message_queue.put(('message', 'Starting to fetch streaming data...'))
    async for line in fetch_streaming_data(url, payload):
        if line:
            try:
                message = json.loads(line)
                if "workflow_id" in message:
                    # Send workflow_id to main thread
                    message_queue.put(('workflow_id', message["workflow_id"]))
                    continue
                event_type = message.get("event")
                if event_type in ["request_user_input", "request_feedback"]:
                    # Send the message to the main thread
                    message_queue.put(('user_input_required', message))
                    # Wait until user input is provided
                    user_input_event.wait()
                    user_input_event.clear()
                    continue
                else:
                    # Send the line to the main thread
                    message_queue.put(('message', message.get('msg', line)))
            except json.JSONDecodeError:
                message_queue.put(('message', line))
        if "[Final result]:" in str(line):
            break  # Stop processing after receiving the final result


def start_long_running_task(url, payload, message_queue, user_input_event):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(get_stream_data(url, payload, message_queue, user_input_event))
        loop.close()
    except Exception as e:
        message_queue.put(('error', f"Exception in background thread: {str(e)}"))


def process_messages():
    while not st.session_state.message_queue.empty():
        try:
            msg_type, content = st.session_state.message_queue.get_nowait()
            if msg_type == 'workflow_id':
                st.session_state.workflow_id = content
            elif msg_type == 'user_input_required':
                st.session_state.user_input_required = True
                st.session_state.user_input_prompt = content
            elif msg_type == 'message':
                st.session_state.received_lines.append(content)
            elif msg_type == 'error':
                st.error(content)
        except queue.Empty:
            pass


def user_input_fragment():
    if st.session_state.user_input_required:
        message = st.session_state.user_input_prompt
        event_type = message.get("event")
        st.write(f"Event received: {event_type}")  # Logging
        if event_type == "request_user_input":
            summary = message.get("summary")
            outline = message.get("outline")
            st.write("Summary:")
            st.write(summary)
            st.write("Outline:")
            st.json(outline)
            user_response = st.radio(
                "Do you approve this outline?",
                ("Yes", "No"),
                key="user_response",
            )
            if st.button("Submit Response", key="submit_response"):
                st.write(f"Submitting response: {user_response}")  # Logging
                # Send the user's response to the backend
                requests.post(
                    "http://backend:80/submit_user_input",
                    json={
                        "workflow_id": st.session_state.workflow_id,
                        "user_input": user_response,
                    },
                )
                st.session_state.user_input_required = False
                st.session_state.user_input_prompt = None
                # Signal the background thread
                st.session_state.user_input_event.set()
        elif event_type == "request_feedback":
            st.write("Please provide your feedback below:")  # Logging
            feedback = st.text_area(message.get("message"), key="user_feedback")
            if st.button("Submit Feedback", key="submit_feedback"):
                st.write(f"Submitting feedback: {feedback}")  # Logging
                requests.post(
                    "http://backend:80/submit_user_input",
                    json={
                        "workflow_id": st.session_state.workflow_id,
                        "user_input": feedback,
                    },
                )
                st.session_state.user_input_required = False
                st.session_state.user_input_prompt = None
                # Signal the background thread
                st.session_state.user_input_event.set()


def main():
    st.title("Slide Generation")

    # Use st_autorefresh to refresh the script every 2 seconds
    st_autorefresh(interval=2000, limit=None, key="data_refresh")

    # Sidebar with form
    with st.sidebar:
        with st.form(key="slide_gen_form"):
            file_dir = st.text_input(
                "Enter the directory path for slide generation:",
                value="./data/summaries_test",
                placeholder="./data/summaries_test",
            )
            submit_button = st.form_submit_button(label="Submit")

    expander_placeholder = st.expander("🤖⚒️Agent is working...")

    if submit_button:
        # Start the long-running task in a separate thread
        if st.session_state.workflow_thread is None or not st.session_state.workflow_thread.is_alive():
            st.write("Starting the background thread...")

            st.session_state.workflow_thread = threading.Thread(
                target=start_long_running_task,
                args=(
                    "http://backend:80/run-slide-gen",
                    {"path": file_dir},
                    st.session_state.message_queue,
                    st.session_state.user_input_event,
                ),
            )
            st.session_state.workflow_thread.start()
            st.session_state.received_lines = []
        else:
            st.write("Background thread is already running.")

    # Process messages from the queue
    process_messages()

    # Check if the thread is alive
    if st.session_state.workflow_thread and st.session_state.workflow_thread.is_alive():
        st.write("Background thread is running.")
    else:
        st.write("Background thread has stopped.")

    # Display received lines
    for line in st.session_state.get('received_lines', []):
        with expander_placeholder:
            st.write(line)

    # Include the user input fragment
    user_input_fragment()


main()
