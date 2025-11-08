import os
import tempfile
import streamlit as st
from langgraph.graph import StateGraph, END
from langgraph.graph.state import State
from openai import OpenAI
from PIL import Image, ImageChops, ImageStat
import requests
from io import BytesIO

# =============== OPENAI SETUP ==================
client = OpenAI(api_key="YOUR_OPENAI_KEY_HERE")  # Replace with your key

# =============== STREAMLIT UI ==================
st.set_page_config(page_title="DesignMatch AI", layout="wide")
st.title("🎨 DesignMatch AI – Figma vs Web Visual QA Agent")

st.sidebar.header("🔧 Input Configuration")
figma_link = st.sidebar.text_input("Enter Figma Link (optional)")
web_url = st.sidebar.text_input("Enter Website URL to Compare")
uploaded_image = st.sidebar.file_uploader("Upload Figma Design Screenshot", type=["png", "jpg", "jpeg"])

run_button = st.sidebar.button("🚀 Run Comparison")

# =============== NODES (LangGraph) ==================

class InputState(State):
    figma_link: str
    web_url: str
    figma_image: Image.Image
    web_image: Image.Image
    diff_image: Image.Image
    ai_report: str


def capture_website_image(state: InputState):
    """Capture website screenshot using requests + simple screenshot service (no browser automation)."""
    try:
        screenshot_url = f"https://image.thum.io/get/{state.web_url}"
        response = requests.get(screenshot_url)
        state.web_image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"Failed to capture website image: {e}")
    return state


def compare_images(state: InputState):
    """Compare Figma design with website image."""
    try:
        figma_img = state.figma_image.resize(state.web_image.size)
        diff = ImageChops.difference(figma_img, state.web_image)
        stat = ImageStat.Stat(diff)
        diff_score = sum(stat.mean) / len(stat.mean)
        state.diff_image = diff
        state.diff_score = diff_score
    except Exception as e:
        st.error(f"Image comparison failed: {e}")
    return state


def ai_analyze(state: InputState):
    """Use GPT to analyze visual differences."""
    try:
        prompt = f"""
        You are a design QA assistant. Analyze visual differences between a Figma design
        and a live web page. The numeric difference score is {round(state.diff_score,2)}.
        Provide insights on what might be wrong (colors, spacing, fonts, layout mismatches).
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        state.ai_report = response.choices[0].message.content
    except Exception as e:
        state.ai_report = f"Error analyzing design: {e}"
    return state


def generate_report(state: InputState):
    """Display visual and textual QA output."""
    st.subheader("📊 Visual Comparison Result")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(state.figma_image, caption="🎨 Figma Design", use_column_width=True)
    with col2:
        st.image(state.web_image, caption="🌐 Website Screenshot", use_column_width=True)
    with col3:
        st.image(state.diff_image, caption=f"🧩 Diff Image (Score: {round(state.diff_score,2)})", use_column_width=True)

    st.subheader("🧠 AI QA Report")
    st.markdown(state.ai_report)
    return state


# =============== BUILD THE GRAPH ==================

graph = StateGraph(InputState)
graph.add_node("capture_website_image", capture_website_image)
graph.add_node("compare_images", compare_images)
graph.add_node("ai_analyze", ai_analyze)
graph.add_node("generate_report", generate_report)

graph.add_edge("capture_website_image", "compare_images")
graph.add_edge("compare_images", "ai_analyze")
graph.add_edge("ai_analyze", "generate_report")
graph.set_entry_point("capture_website_image")
graph.set_finish_point("generate_report")

workflow = graph.compile()

# =============== RUN PIPELINE ==================

if run_button and uploaded_image:
    st.info("Processing comparison…")
    figma_img = Image.open(uploaded_image).convert("RGB")

    initial_state = InputState(figma_link=figma_link, web_url=web_url, figma_image=figma_img)
    workflow.invoke(initial_state)
