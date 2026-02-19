"""Minimal test for browser.py"""
import streamlit as st

st.title("Test")

st.write("If you see this, streamlit works")

try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    st.write("Path inserted")

    from cog_memory.query_interface import CognitiveMemory
    st.write("CognitiveMemory imported")

    memory = CognitiveMemory()
    st.write("Memory created")

    st.success("Everything works!")
except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())
