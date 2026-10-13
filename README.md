# 🍽️ Zomato Restaurant Recommendation System

### Content-Based Restaurant Recommendations Using NLP & Machine Learning

A restaurant recommendation system built using **Natural Language Processing (NLP), TF-IDF, and Cosine Similarity** to recommend restaurants based on customer reviews and ratings.

The project includes an interactive, Zomato-inspired **Streamlit web application**, allowing users to discover restaurants with similar customer experiences.

**[🚀 Try the Live Application](https://zomato-recommendation-system-by-nakul-gupta.streamlit.app/)**

---

## 📌 Project Overview

Finding restaurants with similar food offerings and customer experiences can be challenging when relying solely on ratings.

This project addresses that challenge by analyzing restaurant customer reviews and identifying restaurants with similar review content.

The recommendation system combines:

- **70% Review Similarity** — Calculated using TF-IDF and Cosine Similarity.
- **30% Customer Rating** — Used to incorporate restaurant quality into the ranking.

The application also handles multiple restaurant locations, excludes the searched restaurant's own outlets, and avoids repeating identical restaurant names in the recommendations.

## 🎯 Project Objectives

- Build a content-based recommendation system using customer reviews.
- Apply NLP techniques to process unstructured review text.
- Convert customer reviews into numerical features using TF-IDF.
- Identify similar restaurants using Cosine Similarity.
- Combine similarity and customer ratings for recommendation ranking.
- Develop and deploy an interactive Streamlit application.

## 📊 Dataset

**Source:** Zomato Bangalore Restaurants Dataset (Kaggle)

| Metric | Value |
|---|---:|
| Original dataset records | 51,717 |
| Original features | 17 |
| Extracted customer reviews | 1,317,680 |
| Unique cleaned outlet-review records | 139,832 |
| Restaurant outlets selected for modeling | 5,128 |
| TF-IDF features | 50,000 |

Restaurants with fewer than five unique qualifying reviews were excluded from the modeling dataset to reduce the influence of insufficient review information.

The original dataset is excluded from the GitHub repository.

## 🧠 Methodology

### 1. Data Preprocessing

- Explored missing values, duplicate restaurant records, and review distributions.
- Extracted individual customer reviews from serialized review lists.
- Normalized review text and removed duplicate reviews within each restaurant outlet.
- Selected outlets with at least five unique qualifying reviews.
- Consolidated customer reviews at the restaurant-outlet level.
- Cleaned and aggregated restaurant ratings.

### 2. NLP Feature Engineering

Applied **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert restaurant reviews into numerical feature vectors.

The TF-IDF configuration includes:

- Maximum features: 50,000
- Unigrams and bigrams
- English stop-word removal
- Minimum document frequency: 2
- Maximum document frequency: 85%
- Sublinear term frequency scaling

### 3. Restaurant Similarity

Used **Cosine Similarity** to measure similarity between restaurant review vectors.

The system identifies a pool of similar restaurants based on the textual patterns present in customer reviews.

### 4. Recommendation Ranking

The recommendation engine combines normalized review similarity and restaurant ratings:

**Recommendation Score = 0.70 × Normalized Similarity + 0.30 × Normalized Rating**

The engine:

1. Identifies the selected restaurant and location.
2. Calculates cosine similarity against other restaurant outlets.
3. Excludes other outlets of the selected restaurant.
4. Selects the 100 most similar candidates.
5. Calculates a combined recommendation score.
6. Removes repeated restaurant names.
7. Returns the highest-ranked recommendations.

The 70:30 weighting is an experimental baseline rather than a statistically optimized parameter.

## 🖥️ Streamlit Application

The application provides:

- Searchable restaurant selection.
- Dynamic location selection for restaurants with multiple outlets.
- Restaurant recommendations based on customer reviews.
- Customer ratings and similarity scores.
- A configurable number of recommendations.
- A responsive, Zomato-inspired interface.
- Downloadable recommendation results.

**Live App:** https://zomato-recommendation-system-by-nakul-gupta.streamlit.app/

## 📈 Model Evaluation

The recommendation system was evaluated through qualitative testing across restaurant categories, including fast food, cafés, ice cream, and quick-service meals.

An additional diagnostic evaluation was conducted using 30 randomly sampled restaurant outlets.

| Evaluation Metric | Result |
|---|---:|
| Restaurants evaluated | 30 |
| Average recommendations returned | 10 |
| Average unique restaurant names | 10 |
| Average cosine similarity | 0.1447 |
| Queries returning fewer than 10 recommendations | 0 |

These results demonstrate successful recommendation generation and exact-name diversity within the sampled queries.

**Note:** The evaluation does not establish recommendation accuracy or user satisfaction. Further validation would require relevance labels or real user feedback.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Pandas | Data processing and analysis |
| Scikit-learn | TF-IDF and Cosine Similarity |
| SciPy | Sparse matrix operations |
| Joblib | Model artifact serialization |
| Streamlit | Interactive web application |
| Git & GitHub | Version control and source hosting |
| Streamlit Community Cloud | Application deployment |

## 📁 Project Structure

```text
zomato-recommendation-system/
├── app.py
├── code.ipynb
├── requirements.txt
├── .gitignore
├── models/
│   ├── restaurant_data.joblib
│   ├── tfidf_matrix.joblib
│   └── tfidf_vectorizer.joblib
└── README.md
```

The original dataset and local Python environment are excluded from version control.

## ⚙️ Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/Nakul-GitH/zomato-recommendation-system.git
cd zomato-recommendation-system
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

**3. Activate the environment**

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Launch the application**

```bash
streamlit run app.py
```

The application will typically be accessible at `http://localhost:8501`.

Python 3.12 is recommended for consistency with the development environment. Compatible dependency versions may be required when loading the saved model artifacts.

## ⚠️ Limitations

- Recommendations depend on historical customer reviews, not live Zomato data.
- Restaurants with insufficient review information are excluded.
- TF-IDF captures lexical similarity rather than deeper semantic meaning.
- Restaurant ratings are aggregated from the available dataset.
- Similar restaurant brands with slightly different names may still appear separately.
- Recommendation weights have not been optimized using labeled relevance data.

## 🔮 Future Improvements

- Explore Sentence Transformers for semantic review similarity.
- Incorporate cuisine, price range, and geographic preferences.
- Improve restaurant brand-name normalization.
- Evaluate rating confidence using review or vote counts.
- Collect user feedback to measure recommendation relevance.
- Compare alternative ranking strategies using labeled evaluation data.

## 👨‍💻 Author

**Nakul Gupta**

Applied Data Science | Machine Learning | Natural Language Processing

**GitHub:** https://github.com/Nakul-GitH

**Live Application:** https://zomato-recommendation-system-by-nakul-gupta.streamlit.app/

---

*This is an independent educational Data Science project created by Nakul Gupta using a historical Zomato dataset. It is not an official Zomato product or affiliated with Zomato.*