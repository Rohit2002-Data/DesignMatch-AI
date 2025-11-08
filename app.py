import os
import tempfile
import streamlit as st
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, ConfigDict
from openai import OpenAI
from PIL import Image, ImageChops, ImageStat
from typing import Optional
from io import BytesIO
import requests


# ----------------------------
# 1️⃣ Define LangGraph State
# ----------------------------
class InputState(BaseModel):
    figma_link: Optional[str] = None
    web_url: Optional[str] = None
    figma_image: Optional[Image.Image] = None
    web_image: Optional[Image.Image] = None
    diff_image: Optional[Image.Image] = None
    diff_score: float = 0.0
    ai_report: str = ""

    # Allow PIL.Image types
    model_config = ConfigDict(arbitrary_types_allowed=True)


# ----------------------------
# 2️⃣ Define LangGraph Nodes
# ----------------------------
def load_images(state: InputState) -> InputState:
    """Load images from uploads or URLs"""
    st.info("Loading design and website images...")

    if state.figma_link and not state.figma_image:
        try:
            response = requests.get(state.figma_link)
            state.figma_image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            st.error(f"Failed to load Figma image: {e}")

    if state.web_url and not state.web_image:
        try:
            response = requests.get(state.web_url)
            state.web_image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            st.error(f"Failed to load Website image: {e}")

    return state


def compare_images(state: InputState) -> InputState:
    """Compute pixel-level diff and similarity score"""
    if not state.figma_image or not state.web_image:
        st.warning("Please provide both Figma and Website images.")
        return state

    st.info("Performing image comparison...")

    # Resize both to smallest common size
    figma_resized = state.figma_image.resize(
        min(state.web_image.size, state.figma_image.size)
    )
    web_resized = state.web_image.resize(
        min(state.web_image.size, state.figma_image.size)
    )

    # Compute pixel difference
    diff = ImageChops.difference(figma_resized, web_resized)
    stat = ImageStat.Stat(diff)
    diff_score = sum(stat.mean) / (len(stat.mean) * 255)  # Normalize 0-1

    # Invert for "similarity"
    similarity = (1 - diff_score) * 100

    state.diff_image = diff
    state.diff_score = round(similarity, 2)

    st.success(f"✅ Visual similarity: {state.diff_score}%")

    return state


def analyze_with_ai(state: InputState) -> InputState:
    """Use OpenAI model to generate feedback report"""
    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))

    prompt = f"""
    You are a design QA assistant.
    Compare a Figma design and a built webpage visually.

    The similarity score is {state.diff_score}%.
    Suggest 3 improvements that can make the webpage closer to the design.
    Write a concise and professional report.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional design QA reviewer."},
                {"role": "user", "content": prompt}
            ],
        )
        state.ai_report = response.choices[0].message.content.strip()
    except Exception as e:
        state.ai_report = f"⚠️ AI analysis failed: {e}"

    return state


# ----------------------------
# 3️⃣ Build the LangGraph Workflow
# ----------------------------
graph = StateGraph(InputState)
graph.add_node("load_images", load_images)
graph.add_node("compare_images", compare_images)
graph.add_node("analyze_with_ai", analyze_with_ai)

graph.set_entry_point("load_images")
graph.add_edge("load_images", "compare_images")
graph.add_edge("compare_images", "analyze_with_ai")
graph.add_edge("analyze_with_ai", END)

app = graph.compile()


# ----------------------------
# 4️⃣ Streamlit Frontend
# ----------------------------
st.set_page_config(page_title="DesignMatch AI", layout="wide")
st.title("🎨 DesignMatch AI — Compare Figma vs Web Builds")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Figma Design")
        figma_link = st.text_input("Figma Image URL (optional)")
        figma_image_file = st.file_uploader("Upload Figma Screenshot", type=["png", "jpg", "jpeg"])

    with col2:
        st.subheader("Website Build")
        web_url = st.text_input("Website Screenshot URL (optional)")
        web_image_file = st.file_uploader("Upload Website Screenshot", type=["png", "jpg", "jpeg"])

    submitted = st.form_submit_button("🔍 Run Comparison")

if submitted:
    state = InputState(figma_link=figma_link, web_url=web_url)

    if figma_image_file:
        state.figma_image = Image.open(figma_image_file).convert("RGB")

    if web_image_file:
        state.web_image = Image.open(web_image_file).convert("RGB")

    final_state = app.invoke(state)

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        if final_state.figma_image:
            st.image(final_state.figma_image, caption="🎨 Figma Design", use_container_width=True)
    with col2:
        if final_state.web_image:
            st.image(final_state.web_image, caption="💻 Website Build", use_container_width=True)

    if final_state.diff_image:
        st.image(final_state.diff_image, caption=f"🧮 Diff Overlay — Similarity: {final_state.diff_score}%")

    if final_state.ai_report:
        st.markdown("### 🧠 AI QA Report")
        st.info(final_state.ai_report)
