# Automatic Keyword Extractor Website

A Streamlit website for the NLP course project by Muhammed Fazal T M (ASI23CA042).
It compares TF-IDF, RAKE, TextRank, YAKE, and KeyBERT.

## Run it on your computer

1. Install Python 3.10 or 3.11.
2. Extract this ZIP file into a folder.
3. Open Command Prompt/Terminal in that folder.
4. Create a virtual environment (recommended):

   Windows:
   ```
   py -m venv .venv
   .venv\Scripts\activate
   ```

   macOS/Linux:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
6. Start the website:
   ```
   streamlit run app.py
   ```
7. Your browser should open the website, usually at http://localhost:8501.

The first install/run can take several minutes because the language model and KeyBERT embedding model need to download.

## Publish it online with Streamlit Community Cloud

1. Sign in to GitHub and create a new repository.
2. Upload `app.py` and `requirements.txt` to the repository root.
3. Open https://share.streamlit.io/ and connect GitHub.
4. Select the repository, branch, and `app.py` as the main file.
5. Click Deploy. Wait for the app to install its requirements.
6. Copy the generated website URL to share with your professor.

## Important notes

- Keep `app.py` and `requirements.txt` in the repository root.
- Do not upload passwords, API keys, or private information.
- KeyBERT can take time to download its model on first startup.
- The scores produced by different algorithms use different scales; compare their rankings and extracted phrases rather than raw scores.
- This project is intended for English text.
