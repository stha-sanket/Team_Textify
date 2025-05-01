import streamlit as st
import pytesseract
from PIL import Image
import os # Import os if needed for tesseract path (see below)

# --- Optional: Explicitly set Tesseract path if needed ---
# Uncomment and set the correct path if Streamlit cannot find tesseract
# Example for Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Example for Linux (usually not needed if installed via package manager):
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
# Example for macOS (if installed via Homebrew):
# pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
# ----------------------------------------------------------

def extract_text(image):
    """Extracts text from a PIL Image object using Tesseract."""
    try:
        # Use nep+eng for Nepali and English; adjust languages as needed
        text = pytesseract.image_to_string(image, config='--oem 3 --psm 6 -l nep+eng')
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        st.error("""
            **TesseractNotFoundError:** The Tesseract OCR engine was not found.
            Please ensure Tesseract is installed on your system and accessible in your PATH.
            Alternatively, you can uncomment and set the `pytesseract.tesseract_cmd` path
            at the beginning of the `1_📄_OCR.py` script.
            See Tesseract installation guides for your operating system.
        """)
        return None # Return None to indicate failure
    except Exception as e:
        st.error(f"An error occurred during OCR processing: {e}")
        return "" # Return empty string for other errors

# --- Streamlit UI ---
st.set_page_config(page_title="Nepali OCR", layout="wide") # Config specific to this page

st.title("📝 Nepali OCR from Image")
st.caption("Extract Nepali and English text directly from an image.")
st.divider()

# Allow image file upload
uploaded_image = st.file_uploader("Upload an Image file", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded_image:
    try:
        image = Image.open(uploaded_image)

        # Display the uploaded image in a column
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, caption=f"Uploaded: {uploaded_image.name}", use_column_width=True)

        with col2:
            st.subheader("Extracted Text")
            with st.spinner("⏳ Processing Image..."):
                extracted_text = extract_text(image)

                if extracted_text is None:
                    # Error message already shown by extract_text
                    pass
                elif extracted_text:
                    st.success("✅ Text Extracted Successfully!")
                    st.text_area("Result", extracted_text, height=400)
                    st.download_button(
                         label="Download Text",
                         data=extracted_text,
                         file_name=f"{os.path.splitext(uploaded_image.name)[0]}_extracted.txt",
                         mime="text/plain"
                     )
                else:
                    st.warning("⚠️ No text could be extracted. The image might be blank, unclear, or contain unsupported characters.")

    except Exception as e:
        st.error(f"❌ An error occurred while opening or processing the image: {str(e)}")
else:
    st.info("👆 Upload an image file to begin OCR.")