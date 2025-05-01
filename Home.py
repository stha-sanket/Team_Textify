import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Document Processing Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar message
st.sidebar.success("👈 Select a tool to begin")

# Title and welcome message
st.title("📄 Welcome to the Document Processing Hub! 🖼️")

# Introduction section
st.markdown("""
Welcome to the **Document Processing Hub**, your one-stop solution for intelligent document analysis. This app integrates powerful tools to help you extract insights and automate document workflows.

### 🔧 Available Tools

#### 1. 📷 Nepali OCR
- Extracts printed text from images containing **Nepali** and **English** scripts.
- Supports scanned documents, photos, and screenshots.
- Powered by robust OCR technology optimized for multilingual recognition.

#### 2. 🧠 Document Classifier
- Automatically identifies the type of document from an image.
- Currently supports: **Citizenship**, **National ID (NID)**, and **PAN Card**.
- Useful for document sorting, onboarding processes, and digital archiving.

### 🚀 How to Use
- Choose a tool from the **left sidebar**.
- Upload your document image.
- Review the results in real time.

---

💡 **Pro Tips**
- Use high-quality, well-lit images for best results.
- For OCR, crop the image to the text region if possible.
- Zoom in or click on results to copy text directly.

📩 **Feedback & Support**
Have suggestions or issues? Reach out to the development team or file an issue on our [GitHub repo](https://github.com/stha-sanket/Team_Textify).

---
""")
