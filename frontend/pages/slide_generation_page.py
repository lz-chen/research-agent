import json
import logging

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


def format_workflow_info(info_json: dict):
    # event_type = info_json.get("event_type")
    event_sender = info_json.get("event_sender")
    event_content = info_json.get("event_content")

    try:
        return f"[{event_sender}]: {event_content.get('message')}"
    except Exception as e:
        logging.warning(f"Error formatting workflow info: {str(e)}")
        return json.dumps(info_json)


async def get_stream_data(url, payload, message_queue, user_input_event):
    message_queue.put(('message', 'Starting to fetch streaming data...'))
    async for data in fetch_streaming_data(url, payload):
        if data:
            try:
                data_json = json.loads(data)
                if "workflow_id" in data_json:
                    # Send workflow_id to main thread
                    message_queue.put(('workflow_id', data_json["workflow_id"]))
                    continue
                event_type = data_json.get("event_type")
                event_sender = data_json.get("event_sender")
                event_content = data_json.get("event_content")
                if event_type in ["request_user_input"]:
                    # Send the message to the main thread
                    message_queue.put(('user_input_required', data_json))
                    # Wait until user input is provided
                    user_input_event.wait()
                    user_input_event.clear()
                    continue
                else:
                    # Send the line to the main thread
                    message_queue.put(('message', format_workflow_info(data_json)))
            except json.JSONDecodeError:  # todo: is this necessary?
                message_queue.put(('message', data))
        if "[Final result]:" in str(data):
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


@st.fragment
def user_input_fragment():
    if st.session_state.user_input_required:
        data = st.session_state.user_input_prompt
        event_type = data.get("event_type")
        st.write(f"Event received: {event_type}")  # Logging
        if event_type == "request_user_input":
            summary = data.get("event_content").get("summary")
            outline = data.get("event_content").get("outline")
            prompt_message = data.get("event_content").get("message", "Please review the outline.")

            st.write("Summary:")
            st.markdown(summary)
            st.write("Outline:")
            st.json(outline)
            st.write(prompt_message)

            sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
            approval = st.feedback("thumbs")
            # Display feedback input (optional)
            feedback = st.text_area(
                "Please provide feedback if you have any:",
                key="user_feedback"
            )

            if st.button("Submit Response", key="submit_response"):
                st.write(f"Submitting approval: {approval}, feedback: {feedback}")  # Logging
                user_response = {
                    "approval": sentiment_mapping[approval],
                    "feedback": feedback
                }
                # Send the user's response to the backend
                response = requests.post(
                    "http://backend:80/submit_user_input",
                    json={
                        "workflow_id": st.session_state.workflow_id,
                        "user_input": json.dumps(user_response),
                    },
                )
                st.write(f"Backend response: {response.status_code}")  # Logging
                st.session_state.user_input_required = False
                st.session_state.user_input_prompt = None
                # Signal the background thread
                st.session_state.user_input_event.set()


def main():
    st.set_page_config(
        page_title="Slide Generation",
        page_icon="🧾",
        layout="wide",
        # initial_sidebar_state="expanded",  # expand side bar (horizontally)
    )
    # st.title("Slide Generation")

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

    # Main view divided into two columns
    left_column, right_column = st.columns(2)

    with left_column:
        st.write("Workflow Executions:")
        expander_placeholder = st.expander("🤖⚒️Agent is working...")

    with right_column:
        st.write("Workflow Artifacts:")
        artifact_render = st.empty()

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
    with artifact_render:
        user_input_fragment()


main()
