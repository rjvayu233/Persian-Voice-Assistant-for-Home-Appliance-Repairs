# Persian Voice Assistant for Home Appliance Repairs 🛠️🎙️

## About the Project 📖

This project is a sophisticated Persian voice assistant designed to
assist users in diagnosing and repairing common home appliances. Users
can simply speak their issues in Persian, and the assistant will guide
them through the troubleshooting process using voice recognition and
text-to-speech capabilities. Built with offline models for privacy and
efficiency, it's powered by advanced AI tools to provide a seamless,
interactive experience. 🚀

Whether you're dealing with a malfunctioning refrigerator, washing
machine, or TV, this assistant analyzes your spoken description,
extracts key details like your name, location, and problem, and even
visualizes your location on a map of Iran. It's ideal for
Persian-speaking users seeking quick, guided repairs without needing
internet access for core functionalities.

### Key Highlights ✨

-   **Voice-Based Interaction**: Speak naturally in Persian to input
    your name, location, and appliance issue.
-   **Offline Capabilities**: Utilizes local models for speech
    recognition (Vosk) and text-to-speech (Facebook's MMS-TTS).
-   **Smart Extraction**: Automatically detects appliance type, problem,
    name, and location from spoken input.
-   **Map Integration**: Displays your province on an interactive map of
    Iran using Folium.
-   **User-Friendly UI**: Built with Streamlit for a clean, responsive
    web interface.
-   **Database-Driven**: Includes a built-in QA database for common
    appliance issues in Persian.

This project demonstrates the power of combining speech AI with
practical applications for everyday problems. It's open-source and ready
for contributions!

## Features 🔧

-   **Speech Recognition**: Powered by Vosk for accurate Persian
    speech-to-text conversion. Supports both large and small models for
    flexibility. 🎤
-   **Text-to-Speech**: Uses Facebook's MMS-TTS model to generate
    natural-sounding Persian responses. 🔊
-   **Information Extraction**:
    -   Extracts user name from introductory phrases.
    -   Identifies Iranian provinces and areas (e.g., Tehran districts
        like Vanak or Karaj areas like Gohardasht) using keyword
        matching.
    -   Detects appliance types (e.g., refrigerator, washing machine)
        and specific issues (e.g., "not cooling" or "noisy").
-   **Interactive Chat History**: Displays conversation bubbles for user
    and assistant messages.
-   **Form Submission**: Generates a summary form with extracted details
    for request registration.
-   **Geospatial Visualization**: Highlights the user's province on a
    choropleth map of Iran using local GeoJSON data.
-   **Error Handling**: Graceful fallbacks for model loading, audio
    recording, and recognition failures.
-   **Custom CSS**: RTL support and styled UI for Persian text.
-   **Reset Options**: Buttons to re-record inputs or start a new
    request.

## Technologies Used 🛡️

This project leverages a stack of powerful open-source libraries and
models:

-   **Python 3.12+**: Core programming language. Python Official Site
-   **Streamlit**: For the web-based UI and interactive components.
    Streamlit Docs
-   **Vosk**: Offline speech recognition for Persian. Vosk GitHub
    -   Models: `vosk-model-fa-0.5` (large) and
        `vosk-model-small-fa-0.42` (small).
-   **Transformers (Hugging Face)**: For text-to-speech with the
    `facebook/mms-tts-fas` model. Hugging Face Model
-   **PyAudio & SoundDevice**: Audio recording and playback.
-   **Folium & Streamlit-Folium**: Interactive maps. Folium GitHub
-   **Pandas**: Data handling for map choropleth.
-   **Torch**: Backend for TTS model (CUDA support if available).
    PyTorch Official Site
-   **Other Libraries**: `wave`, `json`, `tempfile`, `re`, `soundfile`,
    `scipy`, `os`.

No internet is required for core voice features, but GeoJSON map data
needs to be downloaded locally.

## Project Structure 📂

Here's an overview of the key files in the repository:

-   `main.py`: The main application script. Handles model loading, audio
    recording/recognition, information extraction, UI rendering, and map
    display.
-   `Persian_voice_assistant.ipynb`: A Jupyter Notebook with
    step-by-step explanations of the code, including code snippets and
    detailed descriptions of each section.
-   `iran-provinces.geojson`: (Recommended download) Local GeoJSON file
    for Iran's provinces map. Download from this source and place in
    your project directory.
-   **Model Folders** (Not included in repo - download separately):
    -   `vosk-model-fa-0.5/`: Large Vosk model for high accuracy.
    -   `vosk-model-small-fa-0.42/`: Small Vosk model for faster
        performance.
-   **Other Dependencies**: Install via `requirements.txt` (generate
    with `pip freeze > requirements.txt`).

## Installation 🛠️

Follow these steps to set up the project locally:

1.  **Clone the Repository**:

        git clone https://github.com/shervinnd/your-repo-name.git
        cd your-repo-name

2.  **Create a Virtual Environment** (Recommended):

        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate

3.  **Install Dependencies**:

        pip install -r requirements.txt

    *Note*: If `requirements.txt` isn't present, install manually:

        pip install streamlit pyaudio vosk transformers torch sounddevice soundfile scipy folium streamlit-folium pandas

4.  **Download Vosk Models**:

    -   Large Model: vosk-model-fa-0.5 - Unzip to `vosk-model-fa-0.5/`.
    -   Small Model: vosk-model-small-fa-0.42 - Unzip to
        `vosk-model-small-fa-0.42/`. Place them in the project root or
        update paths in `main.py`.

5.  **Download MMS-TTS Model**:

    -   The Transformers library will auto-download
        `facebook/mms-tts-fas` on first run.

6.  **Download GeoJSON for Map**:

        wget https://raw.githubusercontent.com/datasets/geo-boundaries-irn/master/iran-provinces.geojson

    Update `GEOJSON_PATH` in `main.py` to point to this file.

7.  **Run the Application**:

        streamlit run main.py

    Open in your browser at `http://localhost:8501`.

*Troubleshooting*: Ensure your microphone is set up. On Linux/Mac, you
may need additional audio libraries (e.g., `portaudio`).

## Usage 📱

1.  **Start the App**: Run via Streamlit and access the web interface.
2.  **Record Name**: Click "🎙️ شروع و ضبط نام (۵ ثانیه)" and speak your
    name (e.g., "سلام، من علی احمدی هستم").
3.  **Record Location**: Speak your province/area (e.g., "تهران، ونک").
4.  **Record Issue**: Describe the problem (e.g., "یخچالم خنک نمیکنه").
5.  **View Summary**: See the extracted form, chat history, and
    highlighted map.
6.  **Reset**: Use buttons to re-record or start anew.

The assistant responds via voice and text. All processing is local for
privacy! 🔒

## Contributing 🤝

Contributions are welcome! Here's how you can help:

1.  Fork the repository.
2.  Create a new branch: `git checkout -b`.
3.  Commit your changes: `git commit -m`.
4.  Push to the branch: `git push origin`.
5.  Open a Pull Request.

Please follow code style guidelines and add tests where possible. Issues
and feature requests are appreciated! 🌟

## License 📄

This project is licensed under the MIT License - see the LICENSE file
for details.

## Acknowledgments 🙏

-   Thanks to Alphacep for Vosk models.
-   Hugging Face for the MMS-TTS model.
-   Streamlit team for the amazing framework.
-   Open-source community for GeoJSON data.

If you find this project useful, star the repo or share it! ⭐

For questions, open an issue or contact me at
\[shervindanesh8282@gmail.com\].

**Powerd By Miracle⚡**
