# Team_Textify: Document Classification and Text Extraction

This application provides a web-based interface for two document processing tasks:
1.  **Nepali/English OCR**: Extracts text from uploaded images containing Nepali and English script using Tesseract OCR.
2.  **Document Classifier**: Predicts the type of an uploaded document image (Birth Certificate, Blank, Citizenship, NID, PAN) using a pre-trained TensorFlow/Keras model (fine-tuned MobileNetV2). It also includes a feedback mechanism to improve future model iterations.

The application is built using Streamlit's native multi-page feature, allowing easy navigation between the tools via a sidebar.

## Features

**📄 OCR Tool:**
*   Upload image files (jpg, jpeg, png, bmp, tiff).
*   Extracts text using Tesseract OCR with support for Nepali (`nep`) and English (`eng`) languages.
*   Displays the uploaded image.
*   Shows the extracted text in a text area.
*   Provides a button to download the extracted text as a `.txt` file.
*   Includes error handling for Tesseract installation issues.

**🖼️ Classifier Tool:**
*   Upload image files (jpg, jpeg, png, bmp).
*   Preprocesses images (resizing, normalization specific to MobileNetV2).
*   Predicts the document class from: 'Birth Certificate', 'Blank', 'Citizenship', 'NID', 'PAN'.
*   Displays the uploaded image.
*   Shows the predicted class and the model's confidence score.
*   Displays confidence scores for all possible classes.
*   Includes a feedback mechanism:
    *   Users can indicate if the prediction was correct or incorrect.
    *   If incorrect, users can select the correct label.
    *   Feedback (timestamp, filename, prediction, confidence, scores, user feedback) is logged to `feedback_log.csv`.
*   Uses Streamlit Session State to maintain prediction results and feedback state during user interaction.
*   Provides an expandable section to view the contents of the `feedback_log.csv`.

## Project Structure
``` bash
my_multipage_app/
├── Home.py # Main landing/welcome page script
├── pages/
│ ├── 1_📄_OCR.py # Script for the OCR tool page
│ └── 2_🖼️_Classifier.py # Script for the Document Classifier tool page
├── my_image_classifier_model.keras # Trained Keras model file for the classifier
├── feedback_log.csv # CSV file where classifier feedback is stored (created automatically)
├── requirements.txt # Python dependencies
└── README.md # This file ```

## Prerequisites

1.  **Python**: Python 3.8 or newer recommended.
2.  **Tesseract OCR Engine**: This is **essential** for the OCR tool.
    *   You must install Tesseract OCR on your operating system.
    *   **Crucially**, ensure the Tesseract executable is added to your system's PATH environment variable OR explicitly set the path in `pages/1_📄_OCR.py` (see commented-out lines for `pytesseract.pytesseract.tesseract_cmd`).
    *   **Language Data**: Ensure you have the necessary Tesseract language data files installed (`nep.traineddata` and `eng.traineddata`). These usually come with the installer or can be downloaded separately and placed in Tesseract's `tessdata` directory.
    *   Installation guides: [Tesseract Wiki](https://github.com/tesseract-ocr/tesseract/wiki)
3.  **Git**: Required if cloning the repository.

## Installation & Setup

1.  **Clone or Download:**
    ```bash
    git clone <your-repository-url> # If using Git
    cd my_multipage_app
    ```

2.  **Place Model File:**
    *   Ensure the pre-trained model file `my_image_classifier_model.keras` is placed directly inside the `my_multipage_app` directory (one level above the `pages` directory).

3.  **Install Tesseract OCR:**
    *   Follow the instructions for your OS (Windows, macOS, Linux) from the [Tesseract Wiki](https://github.com/tesseract-ocr/tesseract/wiki).
    *   Verify the installation and language packs (`nep`, `eng`).
    *   If Tesseract is not in your PATH, uncomment and edit the `pytesseract.pytesseract.tesseract_cmd` line in `pages/1_📄_OCR.py` with the correct path to your `tesseract.exe` (Windows) or `tesseract` executable (Linux/macOS).

4.  **Create Python Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # Windows:
    .\venv\Scripts\activate
    # Linux/macOS:
    source venv/bin/activate
    ```

5.  **Install Python Dependencies:**
    *   Make sure you have a `requirements.txt` file in the `my_multipage_app` directory with the following content:
      ```txt
      streamlit
      tensorflow
      pandas
      Pillow
      pytesseract
      # Add any other specific versions if needed, e.g., tensorflow==2.10.0
      ```
    *   Install the requirements:
      ```bash
      pip install -r requirements.txt
