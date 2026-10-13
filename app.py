import html
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------------------------------------------------
# Application configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Discover Your Next Favorite | Zomato Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
ZOMATO_RED = "#E23744"


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --z-red: #E23744;
        --z-red-dark: #C82435;
        --ink: #1C1C1C;
        --muted: #696969;
        --line: #EDE8E6;
        --paper: #FFFCFA;
    }
    html, body, [class*="css"], [data-testid="stApp"] {
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #FFF8F7 0px, #FFFFFF 430px);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    #MainMenu, footer {visibility: hidden;}
    h1, h2, h3, h4 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        letter-spacing: -0.035em;
    }
    .brand-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 0 28px;
        border-bottom: 1px solid #F2E8E7;
        margin-bottom: 32px;
    }
    .brand-name {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--z-red);
        font-size: 29px;
        font-weight: 800;
        letter-spacing: -1.7px;
    }
    .brand-caption {
        color: #847875;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.7px;
        text-transform: uppercase;
    }
    .hero {
        padding: 32px 0 26px;
        max-width: 860px;
    }
    .eyebrow {
        color: var(--z-red);
        font-size: 12px;
        letter-spacing: 2px;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 14px;
    }
    .hero h1 {
        font-size: clamp(36px, 5.3vw, 62px);
        font-weight: 800;
        line-height: 1.12;
        margin: 0 0 17px;
        color: var(--ink);
    }
    .hero h1 span { color: var(--z-red); }
    .hero p {
        color: #66605F;
        font-size: 17px;
        line-height: 1.7;
        max-width: 700px;
        margin: 0;
    }
    .search-heading {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800;
        font-size: 20px;
        margin: 12px 0 8px;
    }
    .search-subheading {
        color: var(--muted);
        font-size: 14px;
        margin-bottom: 17px;
    }
    div[data-testid="stForm"] {
        border: 1px solid #F0E6E4;
        background: #FFFFFF;
        border-radius: 20px;
        padding: 22px 25px 15px;
        box-shadow: 0 10px 35px rgba(90, 39, 42, .055);
    }
    .stButton > button[kind="primary"],
    div[data-testid="stFormSubmitButton"] button {
        background: var(--z-red);
        color: white;
        border: 1px solid var(--z-red);
        border-radius: 10px;
        font-weight: 700;
        min-height: 44px;
        transition: background .15s ease;
    }
    .stButton > button[kind="primary"]:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background: var(--z-red-dark);
        border-color: var(--z-red-dark);
        color: white;
    }
    div[data-baseweb="select"] > div {
        border-radius: 10px;
    }
    .section-heading {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 25px;
        font-weight: 800;
        color: var(--ink);
        margin: 36px 0 5px;
    }
    .section-description {
        color: var(--muted);
        font-size: 14px;
        margin: 0 0 20px;
    }
    .restaurant-card {
        background: white;
        border: 1px solid #F0E9E7;
        border-radius: 17px;
        padding: 21px 21px 19px;
        margin-bottom: 14px;
        box-shadow: 0 5px 22px rgba(44, 29, 27, .045);
        min-height: 204px;
    }
    .card-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 13px;
    }
    .card-rank {
        color: var(--z-red);
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.1px;
        text-transform: uppercase;
    }
    .rating-pill {
        background: #E9F7ED;
        color: #18763D;
        border-radius: 7px;
        padding: 5px 9px;
        font-weight: 800;
        font-size: 13px;
        white-space: nowrap;
    }
    .restaurant-name {
        color: #24201F;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 18px;
        line-height: 1.35;
        font-weight: 800;
        margin-bottom: 9px;
        overflow-wrap: anywhere;
    }
    .restaurant-location {
        color: #77706F;
        font-size: 13px;
        margin-bottom: 20px;
        min-height: 18px;
        overflow-wrap: anywhere;
    }
    .card-bottom {
        display: flex;
        justify-content: space-between;
        gap: 8px;
        border-top: 1px solid #F2EDEC;
        padding-top: 13px;
        color: #6B6563;
        font-size: 12px;
    }
    .card-bottom strong { color: #3B3432; }
    .stat-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 20px;
    }
    .stat-chip {
        padding: 11px 15px;
        background: #FFF2F0;
        border: 1px solid #FCE4E1;
        border-radius: 10px;
        color: #6B3C3B;
        font-size: 13px;
    }
    .stat-chip strong { color: var(--z-red); }
    .info-panel {
        background: #FFF9F8;
        border: 1px solid #F4E6E3;
        border-radius: 17px;
        padding: 22px 25px;
        color: #635C5A;
        font-size: 14px;
        line-height: 1.8;
    }
    .info-panel strong { color: #302928; }
    .app-footer {
        margin-top: 46px;
        padding-top: 22px;
        border-top: 1px solid #F0E9E7;
        color: #918785;
        font-size: 12px;
        text-align: center;
        line-height: 1.7;
    }
    @media (max-width: 640px) {
        .block-container { padding-top: 1rem; }
        .brand-bar { padding-bottom: 18px; margin-bottom: 16px; }
        .brand-name { font-size: 25px; }
        .brand-caption { font-size: 9px; letter-spacing: .6px; }
        .hero { padding-top: 14px; }
        .hero p { font-size: 15px; }
        div[data-testid="stForm"] { padding: 16px 14px 10px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Load trained artifacts (once per Streamlit process)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading restaurant recommendation model...")
def load_artifacts():
    data = joblib.load(MODEL_DIR / "restaurant_data.joblib")
    matrix = joblib.load(MODEL_DIR / "tfidf_matrix.joblib")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")

    if len(data) != matrix.shape[0]:
        raise ValueError("Restaurant records and TF-IDF vectors are not aligned.")
    if matrix.shape[1] != len(vectorizer.vocabulary_):
        raise ValueError("TF-IDF feature dimensions do not match the vectorizer.")
    if not data.index.equals(pd.RangeIndex(len(data))):
        raise ValueError("Restaurant data must have a sequential index.")
    if data.duplicated(subset=["name", "location"]).any():
        raise ValueError("Restaurant outlet identities are not unique.")
    return data, matrix, vectorizer


try:
    restaurant_data, tfidf_matrix, tfidf_vectorizer = load_artifacts()
except (FileNotFoundError, OSError, ValueError, ImportError) as error:
    st.error("The recommendation model could not be loaded. Verify the models folder and installed package versions.")
    with st.expander("Technical details"):
        st.code(str(error))
    st.stop()


# -----------------------------------------------------------------------------
# Recommendation engine: matches notebook's validated 70/30 approach
# -----------------------------------------------------------------------------
def recommend_restaurants(
    restaurant_name,
    location=None,
    top_n=10,
    candidate_count=100,
    similarity_weight=0.7,
):
    matching = restaurant_data.loc[
        restaurant_data["name"].str.casefold() == restaurant_name.strip().casefold()
    ]
    if matching.empty:
        return pd.DataFrame()

    if location is not None:
        matching = matching.loc[
            matching["location"].str.casefold() == location.strip().casefold()
        ]
    if matching.empty or (location is None and len(matching) > 1):
        return pd.DataFrame()

    selected_index = int(matching.index[0])
    scores = cosine_similarity(tfidf_matrix[selected_index], tfidf_matrix).ravel()

    same_name = (
        restaurant_data["name"].str.casefold()
        == restaurant_data.at[selected_index, "name"].casefold()
    )
    scores[same_name.to_numpy()] = -1.0

    # Only positive-similarity alternatives are eligible.
    eligible_indices = np.flatnonzero(scores > 0)
    if eligible_indices.size == 0:
        return pd.DataFrame()

    ranked_indices = eligible_indices[np.argsort(scores[eligible_indices])[::-1]]
    candidate_indices = ranked_indices[:candidate_count]

    candidates = restaurant_data.iloc[candidate_indices].copy()
    candidates["similarity"] = scores[candidate_indices]
    candidates = candidates.dropna(subset=["average_rating"])
    if candidates.empty:
        return pd.DataFrame()

    max_similarity = candidates["similarity"].max()
    candidates["similarity_normalized"] = candidates["similarity"] / max_similarity
    candidates["rating_normalized"] = candidates["average_rating"] / 5.0
    candidates["recommendation_score"] = (
        similarity_weight * candidates["similarity_normalized"]
        + (1.0 - similarity_weight) * candidates["rating_normalized"]
    )

    candidates = candidates.sort_values(
        ["recommendation_score", "similarity"], ascending=[False, False]
    )
    candidates["normalized_name"] = candidates["name"].str.casefold().str.strip()
    candidates = candidates.drop_duplicates(subset=["normalized_name"], keep="first")

    return candidates[
        ["name", "location", "average_rating", "max_votes", "similarity", "recommendation_score"]
    ].head(top_n).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Header and hero
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="brand-bar">
      <div class="brand-name">zomato<span style="color:#FF9D9D">.</span></div>
      <div class="brand-caption">Restaurant discovery · AI-powered</div>
    </div>
    <div class="hero">
      <div class="eyebrow">✦ A smarter way to discover food</div>
      <h1>Great food deserves<br><span>great recommendations.</span></h1>
      <p>Love a restaurant? Find your next favorite. Explore places with similar
      customer reviews, thoughtfully ranked using real restaurant ratings.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="stat-strip">'
    f'<div class="stat-chip"><strong>{len(restaurant_data):,}</strong> restaurant outlets</div>'
    '<div class="stat-chip"><strong>Bengaluru</strong> restaurant dataset</div>'
    '<div class="stat-chip"><strong>Review-powered</strong> recommendations</div>'
    '</div>',
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Restaurant selection
# -----------------------------------------------------------------------------
st.markdown('<div class="section-heading">Find restaurants like the ones you love</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">Choose a restaurant and its location to discover similar places.</div>',
    unsafe_allow_html=True,
)

restaurant_names = sorted(restaurant_data["name"].dropna().unique(), key=str.casefold)

with st.container():
    selected_name = st.selectbox(
        "Restaurant name",
        options=restaurant_names,
        index=None,
        placeholder="Search from available restaurants...",
        help="Start typing to find a restaurant in the Bengaluru dataset.",
    )

    if selected_name:
        available_locations = sorted(
            restaurant_data.loc[
                restaurant_data["name"] == selected_name, "location"
            ].dropna().unique(),
            key=str.casefold,
        )
    else:
        available_locations = []

    selected_location = st.selectbox(
        "Restaurant location",
        options=available_locations,
        index=0 if len(available_locations) == 1 else None,
        placeholder="Select a location..." if selected_name else "Choose a restaurant first...",
        disabled=not selected_name,
    )

    requested_count = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=15,
        value=10,
        step=1,
    )
    submitted = st.button(
        "Discover similar restaurants  →", type="primary", use_container_width=True
    )


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
if submitted:
    if not selected_name:
        st.warning("Please select a restaurant to continue.")
    elif not selected_location:
        st.warning("Please select the restaurant location to continue.")
    else:
        with st.spinner("Finding your next favorite restaurants..."):
            recommendations = recommend_restaurants(
                restaurant_name=selected_name,
                location=selected_location,
                top_n=requested_count,
            )
        st.session_state["last_search"] = {
            "name": selected_name,
            "location": selected_location,
            "results": recommendations,
        }

if "last_search" in st.session_state:
    result_state = st.session_state["last_search"]
    results = result_state["results"]

    if results.empty:
        st.info("We couldn't find rated, similar alternatives for this restaurant. Try another outlet.")
    else:
        st.markdown(
            '<div class="section-heading">Your next favorites, curated</div>',
            unsafe_allow_html=True,
        )
        searched_name = html.escape(result_state["name"])
        searched_location = html.escape(result_state["location"])
        st.markdown(
            f'<div class="section-description">{len(results)} recommendations inspired by '
            f'<strong>{searched_name}</strong> · {searched_location}</div>',
            unsafe_allow_html=True,
        )

        cols = st.columns(2, gap="medium")
        for rank, (_, item) in enumerate(results.iterrows(), start=1):
            safe_name = html.escape(str(item["name"]))
            safe_location = html.escape(str(item["location"]))
            rating = float(item["average_rating"])
            similarity = float(item["similarity"])
            votes = item["max_votes"]
            votes_label = f'{int(votes):,}' if pd.notna(votes) else "Not available"
            with cols[(rank - 1) % 2]:
                st.markdown(
                    f"""
                    <div class="restaurant-card">
                      <div class="card-top">
                        <span class="card-rank">Recommendation {rank:02d}</span>
                        <span class="rating-pill">★ {rating:.1f} / 5</span>
                      </div>
                      <div class="restaurant-name">{safe_name}</div>
                      <div class="restaurant-location">📍 {safe_location}, Bengaluru</div>
                      <div class="card-bottom">
                        <span>Review similarity <strong>{similarity:.3f}</strong></span>
                        <span>Recorded votes <strong>{votes_label}</strong></span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        csv_data = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="↓ Export recommendations as CSV",
            data=csv_data,
            file_name="zomato_recommendations.csv",
            mime="text/csv",
            use_container_width=False,
        )


# -----------------------------------------------------------------------------
# Explainability and attribution
# -----------------------------------------------------------------------------
st.markdown('<div class="section-heading">Behind the recommendations</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">A transparent look at how your suggestions are generated.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="info-panel">
      <strong>01 · Understand customer reviews.</strong>
      Customer review text is represented using TF-IDF features, including individual words and word pairs.<br>
      <strong>02 · Find similar restaurants.</strong>
      Cosine similarity identifies restaurants with related review language. The selected restaurant's other outlets are excluded.<br>
      <strong>03 · Balance relevance and quality.</strong>
      Among the 100 most similar candidate outlets, recommendations are ranked using
      <strong>70% normalized review similarity + 30% normalized average rating</strong>.
      Only one outlet per recommended restaurant name is shown.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-footer">
      Independent educational Data Science capstone project created by Nakul Gupta· Based on a historical Bengaluru Zomato dataset.<br>
      Ratings, votes, and locations reflect the dataset, not live restaurant availability.
      Zomato-inspired styling; not affiliated with or endorsed by Zomato.
    </div>
    """,
    unsafe_allow_html=True,
)
