# BDC Internal Selection: IELTS Essay Automated Scoring with Transformers

## Overview
This project builds a **machine learning system** to automatically score IELTS essays across four official assessment categories:

1. **Task Achievement**
2. **Coherence and Cohesion**
3. **Lexical Resource**
4. **Grammatical Range and Accuracy**

Instead of manually grading hundreds of essays, this model uses **transformer-based NLP** to understand the meaning, structure, and style of writing — just like a human examiner.

---

## Why This Matters
Essay grading is **time-consuming** and **subjective**.  
By using NLP, we can:
- Provide **instant feedback** to students.
- Reduce **human bias** in grading.
- Allow teachers to focus on **teaching**, not repetitive scoring.

---

## Approach
We experimented with two paths:

| Approach                     | Description |
|------------------------------|-------------|
| **Baseline**                 | TF-IDF features + classical ML models (LightGBM, XGBoost, Random Forest) |
| **Final Model**              | Transformer embeddings (BERT / DistilBERT) + numeric essay statistics |

**Extra features**:
- Essay word count & average word length.
- Outlier removal for extreme essay lengths (<50 words or >700 words).
- Median imputation for missing scores.

**Evaluation Metric**:  
📏 **Mean Squared Error (MSE)** — lower is better.

---

## Workflow
1. **Data Cleaning** – handle missing values, remove unrealistic essays.
2. **Feature Engineering** – extract text statistics, prepare transformer inputs.
3. **Modeling** – fine-tune transformers for multi-output regression.
4. **Validation** – 5-fold cross validation to ensure stable results.
5. **Prediction** – generate final `submission.csv` for test data.

---

## Results
- **Transformer + numeric features** consistently achieved lower MSE than all baseline models.
- K-Fold CV reduced overfitting and improved score stability.
- The model generalizes well for essays in different topics.

---

## How to Use
```bash
# Clone the repo
git clone https://github.com/yourusername/ielts-essay-scoring.git
cd ielts-essay-scoring

# Install dependencies
pip install -r requirements.txt

# Open Jupyter Notebook
jupyter notebook BDC_Internal_ITS_2025.ipynb
