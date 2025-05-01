import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import numpy as np
from PIL import Image
import os
import datetime
import pandas as pd
MODEL_PATH = 'pages/my_image_classifier_model.keras'

IMG_HEIGHT = 224
IMG_WIDTH = 224
FEEDBACK_LOG_FILE = 'feedback_log.csv'
CLASS_NAMES = [
    'Birth Certificate',
    'Blank',
    'Citizenship',
    'NID',
    'PAN'
]

@st.cache_resource
def load_keras_model(model_path):
    """Loads the Keras model from the specified path."""
    if not os.path.exists(model_path):
         # Try resolving relative to the script first (less likely needed now)
         script_dir = os.path.dirname(__file__)
         abs_path_script_relative = os.path.abspath(os.path.join(script_dir, model_path))
         abs_path_cwd_relative = os.path.abspath(model_path)

         st.warning(f"Model not found at relative path '{model_path}'. Trying script-relative: '{abs_path_script_relative}' and CWD-relative: '{abs_path_cwd_relative}'")

         if os.path.exists(abs_path_script_relative):
             model_path = abs_path_script_relative
             print(f"Using script-relative path: {model_path}")
         elif os.path.exists(abs_path_cwd_relative):
             model_path = abs_path_cwd_relative
             print(f"Using CWD-relative path: {model_path}")
         else:
              st.error(f"Model file not found at any checked location.")
              print(f"Model file not found at: {model_path}, {abs_path_script_relative}, or {abs_path_cwd_relative}")
              return None

    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from: {model_path}") # Log to console
        st.success(f"✅ Model loaded successfully from: {os.path.basename(model_path)}") # Show success in UI
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        print(f"Error loading model: {e}") # Log to console
        return None


# --- Image Preprocessing ---
def preprocess_image(image_pil):
    """Preprocesses a PIL image for the model."""
    try:
        # Resize the image
        img_resized = image_pil.resize((IMG_WIDTH, IMG_HEIGHT))

        # Convert PIL image to NumPy array
        img_array = np.array(img_resized)

        # Ensure image is RGB (discard alpha channel if present)
        if img_array.ndim == 3 and img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
        # Handle grayscale images by converting to RGB
        elif img_array.ndim == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        # Handle unexpected shapes
        elif img_array.ndim != 3 or img_array.shape[-1] != 3:
             st.error(f"Unexpected image shape after initial processing: {img_array.shape}. Please upload an RGB or Grayscale image.")
             print(f"Preprocessing Error: Unexpected shape {img_array.shape}")
             return None


        # Add batch dimension (model expects batch of images)
        img_batch = np.expand_dims(img_array, axis=0)

        # Preprocess using the appropriate function for your model's base
        # Ensure the input is float32 as expected by many tf.keras models
        # ** CRITICAL: Use the correct preprocess_input for your base model **
        img_preprocessed = preprocess_input(tf.cast(img_batch, tf.float32))

        return img_preprocessed
    except Exception as e:
        st.error(f"Error during image preprocessing: {e}")
        print(f"Preprocessing Error: {e}")
        return None


# --- Feedback Logging ---
def log_feedback(timestamp, filename, predicted_class, confidence, scores_dict, was_correct, correct_label=None):
    """Logs the feedback information."""
    log_entry = {
        "Timestamp": timestamp,
        "Filename": filename,
        "PredictedClass": predicted_class,
        "Confidence": f"{confidence:.2f}%",
        "WasCorrect": was_correct,
        "UserCorrectLabel": correct_label if correct_label else "N/A",
        # Include all scores for potential analysis later
        **{f"Score_{name}": score_val for name, score_val in scores_dict.items()}
    }
    print("Feedback Received:", log_entry) # Log to console

    # Optional: Append to CSV file
    try:
        df_entry = pd.DataFrame([log_entry])
        # Ensure consistent column order if file exists
        log_file_path = FEEDBACK_LOG_FILE # Use the constant
        file_exists = os.path.exists(log_file_path)
        is_empty = file_exists and os.path.getsize(log_file_path) == 0

        if file_exists and not is_empty:
             try:
                 # Read existing headers to maintain order and add new columns if needed
                 existing_df = pd.read_csv(log_file_path, nrows=0)
                 all_cols = existing_df.columns.union(df_entry.columns, sort=False)
                 df_entry = df_entry.reindex(columns=all_cols) # Align columns
                 df_entry.to_csv(log_file_path, mode='a', index=False, header=False)
             except pd.errors.EmptyDataError: # Handle case where file exists but is empty (redundant with is_empty check but safe)
                 df_entry.to_csv(log_file_path, index=False, header=True)
             except Exception as read_err: # Catch other potential reading errors
                 st.warning(f"Could not properly read existing feedback log headers: {read_err}. Appending with new headers if needed.")
                 df_entry.to_csv(log_file_path, mode='a', index=False, header=not file_exists or is_empty)
        else:
             df_entry.to_csv(log_file_path, index=False, header=True)


    except Exception as e:
        st.warning(f"Could not write feedback to {FEEDBACK_LOG_FILE}: {e}")

    return log_entry # Return the entry for potential display

# --- Initialize Session State (Keys specific to this page) ---
# Using prefixes helps avoid collisions if other pages use similar state keys
prefix = "classifier_"
if f'{prefix}prediction_made' not in st.session_state:
    st.session_state[f'{prefix}prediction_made'] = False
if f'{prefix}predicted_class_name' not in st.session_state:
    st.session_state[f'{prefix}predicted_class_name'] = None
if f'{prefix}confidence' not in st.session_state:
    st.session_state[f'{prefix}confidence'] = 0.0
if f'{prefix}scores_dict' not in st.session_state: # Store the full score dictionary
    st.session_state[f'{prefix}scores_dict'] = {}
if f'{prefix}image_filename' not in st.session_state:
    st.session_state[f'{prefix}image_filename'] = ""
if f'{prefix}feedback_submitted' not in st.session_state:
    st.session_state[f'{prefix}feedback_submitted'] = False
if f'{prefix}feedback_message' not in st.session_state:
     st.session_state[f'{prefix}feedback_message'] = None # Use None to check if a message exists
if f'{prefix}current_filename' not in st.session_state: # Tracks the name of the file currently processed
     st.session_state[f'{prefix}current_filename'] = None

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Document Classifier") # Config specific to this page

st.title("🖼️ Document Image Classifier with Feedback")
st.write("Upload an image of a document. The model will predict its type. Please provide feedback on the prediction!")
st.markdown(f"**Model Base:** MobileNetV2 (Fine-tuned)") # Or mention your specific base model
st.divider()

# Load the model
# Use the MODEL_PATH defined earlier
model = load_keras_model(MODEL_PATH)

if model is None:
    st.error(f"Model could not be loaded. Please ensure the model file exists at the expected path and is a valid Keras model.")
    st.stop() # Stop execution of this page if model fails to load
# else: # Model loaded successfully, continue with the rest of the page UI

# File uploader
uploaded_file = st.file_uploader(
    "Choose a document image...",
    type=["jpg", "jpeg", "png", "bmp"],
    key=f"{prefix}file_uploader" # Unique key using prefix
)

# --- State Reset Logic (Using prefixed state keys) ---
new_file_uploaded = False
if uploaded_file is not None:
    if uploaded_file.name != st.session_state[f'{prefix}current_filename']:
        print(f"Classifier: New file detected: {uploaded_file.name}. Resetting state.")
        new_file_uploaded = True
        st.session_state[f'{prefix}current_filename'] = uploaded_file.name
        # Reset flags for the new file
        st.session_state[f'{prefix}prediction_made'] = False
        st.session_state[f'{prefix}feedback_submitted'] = False
        st.session_state[f'{prefix}feedback_message'] = None
        # Clear previous prediction results
        st.session_state[f'{prefix}predicted_class_name'] = None
        st.session_state[f'{prefix}confidence'] = 0.0
        st.session_state[f'{prefix}scores_dict'] = {}
        st.session_state[f'{prefix}image_filename'] = uploaded_file.name # Store new filename
elif st.session_state[f'{prefix}current_filename'] is not None:
    # File was removed
    print("Classifier: File removed. Resetting state.")
    st.session_state[f'{prefix}current_filename'] = None
    st.session_state[f'{prefix}prediction_made'] = False
    st.session_state[f'{prefix}feedback_submitted'] = False
    st.session_state[f'{prefix}feedback_message'] = None
    st.session_state[f'{prefix}predicted_class_name'] = None
    st.session_state[f'{prefix}confidence'] = 0.0
    st.session_state[f'{prefix}scores_dict'] = {}
    st.session_state[f'{prefix}image_filename'] = ""


# --- Main Processing Area ---
if uploaded_file is not None:
    col1, col2 = st.columns([0.6, 0.4]) # Adjust column width ratio

    with col1:
        st.subheader("Uploaded Image")
        try:
            # Open the image file
            image = Image.open(uploaded_file)
            st.image(image, caption=f'Uploaded: {st.session_state[f"{prefix}image_filename"]}', width=500) # Use filename from state
        except Exception as e:
            st.error(f"Error opening or displaying image: {e}")
            # If image fails to load, prevent further processing
            st.session_state[f'{prefix}prediction_made'] = False
            st.session_state[f'{prefix}feedback_submitted'] = False
            image = None # Ensure image is None

    # Proceed only if image was loaded successfully
    if image:
        with col2:
            st.subheader("Prediction Results")

            # --- Prediction Logic ---
            # Run prediction only if it hasn't been made for this file yet
            if not st.session_state[f'{prefix}prediction_made']:
                with st.spinner('🧠 Analyzing image...'):
                    processed_image = preprocess_image(image)

                    if processed_image is not None:
                        try:
                            prediction = model.predict(processed_image)
                            score = tf.nn.softmax(prediction[0]).numpy() # Ensure probabilities

                            predicted_class_index = np.argmax(score)
                            predicted_class_name = CLASS_NAMES[predicted_class_index]
                            confidence = np.max(score) * 100
                            scores_dict = {name: f"{prob*100:.2f}%" for name, prob in zip(CLASS_NAMES, score)}

                            # Store results in session state (using prefixed keys)
                            st.session_state[f'{prefix}predicted_class_name'] = predicted_class_name
                            st.session_state[f'{prefix}confidence'] = confidence
                            st.session_state[f'{prefix}scores_dict'] = scores_dict
                            st.session_state[f'{prefix}prediction_made'] = True # Mark prediction as done
                            st.session_state[f'{prefix}feedback_submitted'] = False # Reset feedback status for new prediction
                            st.session_state[f'{prefix}feedback_message'] = None # Clear previous feedback message

                            print(f"Classifier: Prediction successful for {st.session_state[f'{prefix}image_filename']}")


                        except Exception as e:
                            st.error(f"An error occurred during prediction: {e}")
                            print(f"Classifier Prediction Error: {e}")
                            # Ensure state reflects failed prediction
                            st.session_state[f'{prefix}prediction_made'] = False
                    else:
                        st.error("Image could not be preprocessed. Cannot predict.")
                        st.session_state[f'{prefix}prediction_made'] = False # Ensure state reflects failed preprocessing

            # --- Display Prediction Results (If prediction was made, use prefixed state keys) ---
            if st.session_state[f'{prefix}prediction_made']:
                # Display the results using values stored in session state
                st.metric(
                    label="Predicted Document Type",
                    value=st.session_state[f'{prefix}predicted_class_name'],
                    delta=f"{st.session_state[f'{prefix}confidence']:.2f}% Confidence"
                )
                st.subheader("Confidence Scores:")
                st.dataframe(pd.Series(st.session_state[f'{prefix}scores_dict'], name="Confidence"), use_container_width=True)


                st.divider() # Separate prediction from feedback

                # --- Feedback Section ---
                st.subheader("Feedback on Prediction")

                # Show confirmation message if feedback was submitted
                if st.session_state[f'{prefix}feedback_submitted'] and st.session_state[f'{prefix}feedback_message']:
                    st.success(st.session_state[f'{prefix}feedback_message'])
                    st.caption("Upload a new image or refresh to provide feedback again for this image.")
                # Show feedback form only if prediction is made and feedback *not yet* submitted
                elif not st.session_state[f'{prefix}feedback_submitted']:
                    feedback_correct = st.radio(
                        "Was this prediction correct?",
                        ("Yes", "No"),
                        key=f"{prefix}feedback_radio_{st.session_state[f'{prefix}image_filename']}", # Unique key per image session
                        index=None # Default to no selection
                    )

                    correct_label_selection = None
                    if feedback_correct == "No":
                        correct_label_selection = st.selectbox(
                            "Please select the correct label:",
                            CLASS_NAMES,
                            index=None, # No default selection
                            placeholder="Select the true document type...",
                            key=f"{prefix}correct_label_{st.session_state[f'{prefix}image_filename']}" # Unique key
                        )

                    submit_feedback = st.button("Submit Feedback", key=f"{prefix}submit_btn_{st.session_state[f'{prefix}image_filename']}")

                    if submit_feedback:
                        # Ensure all necessary info is available from session state (using prefixed keys)
                        pred_name = st.session_state.get(f'{prefix}predicted_class_name')
                        scores = st.session_state.get(f'{prefix}scores_dict')
                        conf = st.session_state.get(f'{prefix}confidence')
                        filename = st.session_state.get(f'{prefix}image_filename')

                        if pred_name and scores and conf is not None and filename:
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            if feedback_correct == "Yes":
                                log_feedback(timestamp, filename, pred_name, conf, scores, True)
                                st.session_state[f'{prefix}feedback_message'] = f"✅ Feedback received: Marked as correct. Thank you!"
                                st.session_state[f'{prefix}feedback_submitted'] = True
                                st.rerun() # Rerun to show the confirmation message

                            elif feedback_correct == "No":
                                if correct_label_selection:
                                    log_feedback(timestamp, filename, pred_name, conf, scores, False, correct_label_selection)
                                    st.session_state[f'{prefix}feedback_message'] = f"📝 Feedback received: Marked as incorrect. Correct label '{correct_label_selection}' logged. Thank you!"
                                    st.session_state[f'{prefix}feedback_submitted'] = True
                                    st.rerun() # Rerun to show the confirmation message
                                else:
                                    st.warning("Please select the correct label before submitting.")
                            else:
                                st.warning("Please select 'Yes' or 'No' before submitting feedback.")
                        else:
                            st.error("Prediction details seem to be missing from session state. Cannot submit feedback.")


# Message when no file is uploaded
elif not st.session_state[f'{prefix}prediction_made']: # Show only if no file is up or prediction hasn't happened
    st.info("👆 Upload an image file using the uploader above to start.")


st.divider()
st.caption("ML Model trained using Transfer Learning. Feedback helps improve future versions!")

# --- Use st.expander to view feedback log ---
log_expander = st.expander("View Feedback Log", expanded=False) # Collapsed by default

with log_expander:
    st.subheader("Feedback Log Entries") # Add a title inside the expander
    log_file_path = FEEDBACK_LOG_FILE # Use the constant
    if os.path.exists(log_file_path):
        try:
            # Add error handling for empty file during read
            if os.path.getsize(log_file_path) > 0:
                 df_feedback = pd.read_csv(log_file_path)
                 # Display only relevant columns for brevity by default, or allow full view
                 # relevant_cols = ["Timestamp", "Filename", "PredictedClass", "Confidence", "WasCorrect", "UserCorrectLabel"]
                 # display_df = df_feedback[relevant_cols] if all(col in df_feedback.columns for col in relevant_cols) else df_feedback
                 st.dataframe(df_feedback, use_container_width=True) # Show all columns
            else:
                 st.info(f"Feedback log file ('{FEEDBACK_LOG_FILE}') exists but is empty.")
        except pd.errors.ParserError as pe: # Catch specific pandas parsing errors
            st.error(f"Error parsing the feedback log file ('{FEEDBACK_LOG_FILE}'): {pe}. The file might be corrupted.")
        except Exception as e:
            st.error(f"Could not read or display feedback log file ('{FEEDBACK_LOG_FILE}'): {e}")
    else:
        st.info(f"No feedback log file found ('{FEEDBACK_LOG_FILE}'). Nothing to display yet.")