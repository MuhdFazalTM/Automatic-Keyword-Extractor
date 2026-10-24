import re
import nltk
import pandas as pd
import streamlit as st
import spacy
import networkx as nx
import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake
from keybert import KeyBERT
from nltk.stem import PorterStemmer

st.set_page_config(page_title="Automatic Keyword Extractor", page_icon="🔎", layout="wide")

@st.cache_resource(show_spinner="Loading language and keyword models...")
def load_models():
    for resource in ("stopwords", "punkt", "punkt_tab"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass
    try:
        nlp_model = spacy.load("en_core_web_sm")
    except OSError:
        st.error("The English spaCy model is missing. Check that requirements.txt was installed correctly.")
        st.stop()
    kb_model = KeyBERT(model="all-MiniLM-L6-v2")
    return nlp_model, kb_model

nlp, keybert_model = load_models()
stemmer = PorterStemmer()

def preprocess_text(text):
    doc = nlp(text.lower())
    return " ".join(t.lemma_ for t in doc if not t.is_stop and not t.is_punct and not t.is_space and t.is_alpha)

def get_candidates(text, max_ngram=3):
    doc = nlp(text)
    candidates = set()
    for chunk in doc.noun_chunks:
        phrase = " ".join(t.lemma_.lower() for t in chunk if not t.is_stop and t.is_alpha)
        if phrase:
            candidates.add(phrase)
    for n in range(1, max_ngram + 1):
        for i in range(len(doc) - n + 1):
            chunk = list(doc[i:i+n])
            if all(t.is_alpha and not t.is_stop for t in chunk):
                candidates.add(" ".join(t.lemma_.lower() for t in chunk))
    return sorted(candidates)

def extract_tfidf(text, top_n=10):
    sentences = [preprocess_text(s.text) for s in nlp(text).sents]
    sentences = [s for s in sentences if s]
    if not sentences:
        return []
    try:
        vec = TfidfVectorizer(ngram_range=(1, 3), max_features=3000)
        mat = vec.fit_transform(sentences)
    except ValueError:
        return []
    scores = mat.sum(axis=0).A1
    return [(w, float(s)) for w, s in sorted(zip(vec.get_feature_names_out(), scores), key=lambda x: x[1], reverse=True)[:top_n]]

def extract_rake(text, top_n=10):
    r = Rake(language="english", min_length=1, max_length=3)
    r.extract_keywords_from_text(text)
    return [(p.lower(), float(s)) for s, p in r.get_ranked_phrases_with_scores()[:top_n]]

def extract_textrank(text, top_n=10):
    doc = nlp(text.lower())
    words = [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop and t.pos_ in {"NOUN", "PROPN", "ADJ"}]
    graph = nx.Graph()
    for a, b in zip(words, words[1:]):
        if a != b:
            if graph.has_edge(a, b):
                graph[a][b]["weight"] += 1
            else:
                graph.add_edge(a, b, weight=1)
    if graph.number_of_nodes() == 0:
        return []
    scores = nx.pagerank(graph, weight="weight")
    ranked = [(p, sum(scores.get(w, 0.0) for w in p.split())) for p in get_candidates(text)]
    return sorted([(p, s) for p, s in ranked if s > 0], key=lambda x: x[1], reverse=True)[:top_n]

def extract_yake(text, top_n=10):
    ex = yake.KeywordExtractor(lan="en", n=3, top=top_n, dedupLim=0.7)
    return [(p.lower(), float(s)) for p, s in ex.extract_keywords(text)]

def extract_keybert(text, top_n=10):
    if not text.strip():
        return []
    return [(p.lower(), float(s)) for p, s in keybert_model.extract_keywords(
        text, keyphrase_ngram_range=(1, 3), stop_words="english",
        top_n=top_n, use_mmr=True, diversity=0.5)]

METHODS = {
    "TF-IDF": extract_tfidf,
    "RAKE": extract_rake,
    "TextRank": extract_textrank,
    "YAKE": extract_yake,
    "KeyBERT": extract_keybert,
}

def extract_all_keywords(text, top_n=10):
    rows = []
    for method, fn in METHODS.items():
        try:
            for rank, (phrase, score) in enumerate(fn(text, int(top_n)), start=1):
                rows.append({
                    "Method": method,
                    "Rank": rank,
                    "Keyword / Keyphrase": phrase,
                    "Score": round(float(score), 4)
                })
        except Exception as e:
            rows.append({
                "Method": method, "Rank": "-", 
                "Keyword / Keyphrase": f"Could not run: {str(e)[:160]}",
                "Score": ""
            })
    return pd.DataFrame(rows, columns=["Method", "Rank", "Keyword / Keyphrase", "Score"])

SAMPLE_TEXT = """Artificial intelligence is transforming the education sector by providing personalized learning experiences for students. Machine learning algorithms analyze student performance, identify learning difficulties, and recommend suitable study materials. Natural Language Processing helps computers understand human language and supports applications such as chatbots, automatic summarization, and keyword extraction. These technologies help teachers save time, improve student engagement, and make education more accessible. However, data privacy, algorithmic bias, and the reliability of AI-generated results remain important challenges."""

st.title("🔎 Automatic Keyword Extractor")
st.write("**An NLP-based course project**")
st.write("Enter an English paragraph or article to extract and compare important keywords using five NLP methods.")

with st.sidebar:
    st.header("Settings")
    top_n = st.slider("Keywords per method", min_value=5, max_value=20, value=10)
    st.caption("The score scale differs between algorithms. Compare the rankings and phrases, not raw scores alone.")

text_input = st.text_area("Paste your text here", value=SAMPLE_TEXT, height=240, placeholder="Paste an English article, abstract, or paragraph...")
run = st.button("Extract Keywords", type="primary", use_container_width=True)

if run:
    if not text_input.strip():
        st.warning("Please paste some text first.")
    elif len(text_input.split()) < 5:
        st.warning("Please enter a little more text (at least five words) for more useful results.")
    else:
        with st.spinner("Extracting keywords with five methods. The first run may take a little longer..."):
            result_df = extract_all_keywords(text_input, top_n)
        st.subheader("Extracted Keywords")
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results as CSV", data=csv, file_name="keyword_extraction_results.csv", mime="text/csv")
        st.subheader("Results by method")
        tabs = st.tabs(list(METHODS.keys()))
        for tab, method in zip(tabs, METHODS.keys()):
            with tab:
                method_df = result_df[result_df["Method"] == method].reset_index(drop=True)
                if method_df.empty:
                    st.info("No keywords were found for this method.")
                else:
                    st.dataframe(method_df[["Rank", "Keyword / Keyphrase", "Score"]], use_container_width=True, hide_index=True)
        st.success("Done! The table contains the keywords, their ranks, and method-specific scores.")

with st.expander("What do the methods mean?"):
    st.markdown("""
    - **TF-IDF:** ranks terms using their frequency in the text and how distinctive they are across text units.
    - **RAKE:** finds candidate phrases and scores them using word relationships and frequency.
    - **TextRank:** ranks words/phrases using connections between terms in a graph.
    - **YAKE:** uses features such as frequency and position to identify keywords. Its scores are generally lower for stronger keywords.
    - **KeyBERT:** uses text embeddings to find phrases semantically related to the document.

    Different methods can return different results. Their raw scores are not directly comparable.
    """)

st.divider()
st.caption("Course project | Muhammed Fazal T M | Register No. ASI23CA042")
