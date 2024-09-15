import streamlit


def set_streamlit_page_config_once():
    try:
        streamlit.set_page_config(layout="wide")
    except streamlit.errors.StreamlitAPIException as e:
        if "can only be called once per app" in e.__str__():
            # ignore this error
            return
        raise e
