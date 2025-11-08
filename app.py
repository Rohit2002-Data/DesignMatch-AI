import os
import streamlit as st
from langgraph.graph import StateGraph
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI
from PIL import Image, ImageChops, ImageStat
import requests
from io import BytesIO


# =====================================================
# 🔑 OPENAI SETUP (replace with your actual key)
# =====================================================
client = OpenAI(api_key="YOUR_OPENAI_KEY_HERE")  # Replace with your key or environment variable


# =====================================================
# 🎨 STREAMLIT UI
# =====================================================
st.set_page_config(page_title="DesignMatch AI", layout="wide")
st.title("🎨 DesignMatch AI – Figma vs Web Visual QA Agent")

st.sidebar.header("🔧 Input Configuration")
figma_link = st.sidebar.text_input("Enter Figma Link (optional)")
web_url = st.sidebar.text_input("Enter Website URL to Compare")
uploaded_image = st.sidebar.file_uploader("Upload Figma Design Screenshot", type=["png", "jpg", "jpeg"])

run_button = st.sidebar.button("🚀 Run Comparison")


# =====================================================
# 🧩 DEFINE STATE CLASS (Pydantic v2 Compatible)
# =====================================================
class InputState(BaseModel):
    figma_link: Optional[str] = None
    web_url: Optional[str] = None
    figma_image: Optional[Image.Image] = None
    web_image: Optional[Image.Image] = None
    diff_image: Optional[Image.Image] = None
    diff_score: float = 0.0
    ai_report: str = ""


# =====================================================
# 🧱 NODES (LangGraph)
# =====================================================

def capture_website_image(state: InputState):
    """Capture website screenshot using Thum.io API (no browser setup needed)."""
    st.info("📸 Capturing website screenshot...")
    try:
        screenshot_url = f"https://image.thum.io/get/{state.web_url}"
        response = requests.get(screenshot_url, timeout=30)
        if response.status_code == 200:
            state.web_image = Image.open(BytesIO(response.content)).convert("RGB")
            st.success("✅ Website screenshot captured successfully!")
        else:
            st.error("❌ Failed to fetch website screenshot.")
    except Exception as e:
        st.error(f"❌ Error capturing website: {e}")
    return state


def compare_images(state: InputState):
    """Compare uploaded Figma design with live website image."""
    st.info("🧩 Comparing images...")
    try:
        figma_img = state.figma_image.resize(state.web_image.size)
        diff = ImageChops.difference(figma_img, state.web_image)
        stat = ImageStat.Stat(diff)
        diff_score = sum(stat.mean) / len(stat.mean)
        state.diff_image = diff
        state.diff_score = diff_score
        st.success(f"✅ Image comparison complete! Difference score: {round(diff_score, 2)}")
    except Exception as e:
        st.error(f"❌ Image comparison failed: {e}")
    return state


def ai_analyze(state: InputState):
    """Use OpenAI GPT model to analyze the visual differences."""
    st.info("🤖 Analyzing visual differences with AI...")
    try:
        prompt = f"""
        You are a design QA assistant.
        The system compared a Figma design with a live webpage.
        The numeric difference score is {round(state.diff_score, 2)} (lower = more similar).

        Based on this score, describe:
        - What visual differences may exist (colors, spacing, font, layout, alignment)
        - What potential HTML/CSS fixes could align them
        - Give an overall match percentage and qualitative assessment
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )

        state.ai_report = response.choices[0].message.content.strip()
        st.success("✅ AI analysis completed successfully!")
    except Exception as e:
        state.ai_report = f"❌ AI analysis failed: {e}"
        st.error(state.ai_report)
    return state


def generate_report(state: InputState):
    """Display all results in Streamlit UI."""
    st.success("🎉 Comparison and analysis complete!")
    st.subheader("📊 Visual Comparison Result")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(state.figma_image, caption="🎨 Figma Design", use_column_width=True)
    with col2:
        st.image(state.web_image, caption="🌐 Website Screenshot", use_column_width=True)
    with col3:
        st.image(state.diff_image, caption=f"🧩 Diff Image (Score: {round(state.diff_score, 2)})", use_column_width=True)

    st.divider()
    st.subheader("🧠 AI QA Report")
    st.markdown(state.ai_report)

    return state


# =====================================================
# 🕸️ BUILD LANGGRAPH WORKFLOW
# =====================================================

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


# =====================================================
# 🚀 RUN THE PIPELINE
# =====================================================

if run_button and uploaded_image:
    st.info("Processing your comparison request...")

    figma_img = Image.open(uploaded_image).convert("RGB")

    initial_state = InputState(
        figma_link=figma_link,
        web_url=web_url,
        figma_image=figma_img,
    )

    workflow.invoke(initial_state)

elif run_button and not uploaded_image:
    st.warning("⚠️ Please upload a Figma design screenshot before running.")
