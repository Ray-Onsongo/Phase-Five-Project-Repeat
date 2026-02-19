# Data-Driven Job Market Analytics Platform
![Project Banner](Images/Project_banner.png)
### DSPT-12

**Project Date:** February 2026  

## Team Members
- Kigen Tuwei
- Kelvin Sesery
- Ray Onsongo 
- Kennedy Wamwati  
- Victor Wasunna

---

## Project Overview

The Job Market Intelligence System is designed to address the fragmented and opaque nature of today’s job market by transforming raw job posting data into actionable, real-time insights for job seekers, HR professionals, and educators. By analyzing English-language technical and professional job postings, the system identifies in-demand and declining skills, maps geographic hiring hotspots, benchmarks salaries by role and experience, and classifies emerging job trends. Its goal is to reduce information overload for job seekers, improve competitive hiring and compensation alignment for recruiters, and enable data-driven curriculum updates for educational institutions. Success will be measured through strong technical performance, such as over 80% job classification `accuracy` and salary prediction `MAE` below $15,000, alongside the delivery of clear, practical insights that support smarter career, hiring, and educational decisions. The Dataset used is [Download the dataset](https://www.kaggle.com/datasets/techsalerator/job-posting-data-in-kenya).

## 1. Business Problem

The modern job market operates in a fragmented and opaque environment where critical information about skill demand, compensation benchmarks, and geographic opportunities is scattered across thousands of job postings. This lack of centralized, data-driven insight creates inefficiencies for **job seekers** making career decisions, **HR professionals** competing for talent, and **educational institutions** attempting to align curricula with market needs. Without a unified intelligence system, stakeholders rely on incomplete or outdated information, leading to misaligned skills, uncompetitive salary offers, prolonged hiring cycles, and graduates entering the workforce unprepared for current demand. 

Some of the **business objectives** are;
- Develop a centralized Job Market Intelligence System that transforms raw job posting data into structured, actionable insights.
- Identify high-demand and declining technical skills to guide career development and curriculum updates.
- Map geographic hiring trends to highlight opportunity hotspots and regional demand shifts.
- Provide data-driven salary benchmarking by role, experience level, and location to improve compensation alignment.
- Classify job postings and detect emerging roles to support strategic workforce and educational planning.

---

## 2. Data Understanding
This project follows the **CRISP-DM (Cross Industry Standard Process for Data Mining)** framework to ensure a structured and rigorous data science workflow.

**Data Source:** 

The data was obtained from **Kaggle**, [The dataset](https://www.kaggle.com/datasets/techsalerator/job-posting-data-in-kenya). Techsalerator's Job Openings Data for Kenya provides a detailed and valuable overview of job opportunities across various sectors in Kenya. This dataset consolidates and categorizes job-related information from multiple sources, including company websites, job boards, and recruitment agencies, offering key insights into the Kenyan labor market.

**Dataset Overview:**

The dataset contains 9,919 job postings collected between February and September 2024, with 21 features describing job titles, descriptions, locations, seniority levels, contract types, salary information, and standardized occupational classifications. Most critical columns such as Job Opening Title (100%), Description (98.9%), Location (95.9%), and Seniority (100%) are highly complete, providing strong analytical reliability.

The dataset is predominantly text-based, making it highly suitable for:
- Skill extraction and demand analysis
- Job classification modeling
- Geographic opportunity mapping
- Market trend analysis

**Data Quality Assesment:**

The dataset demonstrates strong completeness in core descriptive fields and is well-suited for NLP-driven skill analysis and job market intelligence. Moderate preprocessing is required for JSON parsing, categorical normalization, and handling sparse salary data before advanced modeling.

Some of the data quality highlights are;
- `Ticker:` 100% missing, we removed it from analysis.
- `Category` 94% missing
- `Salary(JSON)` Structured but requires parsing
- `Location(JSON)` Requires parsing

**Data Cleaning:**

From our observaions, we noted that there were issues we needed to tackle so as to get the data ready for modelling. We decided to tackle the issues in this order;
- Drop completely empty columns
- Parse JSON columns (Location and Salary Data)
- Handle missing values
- Convert date columns
- Clean categorical/text data
- Create new features for the model

---

## 3. Feature Engineering

This project implements a structured and systematic feature engineering pipeline designed to transform raw job posting data into a model-ready dataset. Following initial exploratory analysis, inconsistencies in categorical fields, particularly within the `Category_list column`, were identified and corrected. Entries stored as malformed strings, empty values, invalid JSON-like structures, or placeholder categories such as “unknown” were standardized to ensure clean and reliable categorical extraction.

Text-based feature engineering was applied to the `Description` column to extract interpretable and lightweight numerical signals before introducing advanced NLP techniques. These engineered features capture description length and complexity, detect requirement-related language, identify educational qualifications, and convert recurring textual patterns into structured indicators. This approach preserves interpretability while improving predictive potential.

Geographical features were engineered to capture broader regional trends without inflating dimensionality. Since country and state fields often exhibit high cardinality, direct encoding would introduce sparsity and overfitting risk. Instead, geographic attributes were grouped, rare categories were handled appropriately, and binary indicators were constructed to reflect meaningful regional patterns.

Company-level features were aggregated to address the high-cardinality nature of raw company names. Rather than encoding each company individually, posting frequency and dominance metrics were computed to capture organizational scale and hiring intensity. This enables the model to learn from company behavior without introducing excessive categorical expansion.

Temporal features were extracted from job posting timestamps to model seasonal, weekly, and recency-related trends. These include quarter-based hiring cycles, weekday versus weekend activity, and time-based signals reflecting hiring urgency. By transforming raw timestamps into structured features, the model can leverage behavioral patterns in job posting activity.

Composite features were constructed to capture higher-order interactions between seniority, job categories, title indicators, and company context. These interaction features quantify role complexity, technical specialization, and seniority-function relationships, providing richer representations beyond isolated variables.

Finally, the modeling pipeline organizes engineered features into logical groups, dynamically selects available predictors, defines balanced target variables, and prepares the dataset for training through encoding and scaling. This structured approach ensures flexibility, scalability, and reproducibility while aligning modeling strategies with the project’s business objectives.

---

## 4. Modelling

The primary modeling objective was **Job Category Classification**, a 23-class multi-class classification task designed to automate job categorization for recruiters. Additional supervised tasks included Seniority Level Prediction, US Job Prediction (binary), and Full-Time Job Prediction (binary). Multiple algorithms were evaluated across these tasks, including Random Forest, Logistic Regression (One-vs-Rest), XGBoost, Gradient Boosting, SVM, and Neural Networks.

A `Dummy Stratified Classifier` established a baseline accuracy of 11.37%, reflecting random class distribution performance. Initial models significantly outperformed this baseline. Random Forest achieved 58.87% accuracy, Logistic Regression reached 54.49%, and XGBoost performed best among the initial models at 60.04% accuracy with stable 5-fold cross-validation (mean ≈ 59.4%).

To improve performance further, several enhancement strategies were implemented:
- Class imbalance handling using inverse-frequency class weights
- Enhanced text feature engineering
- Creation of interaction (composite) features
- Model stacking
- Hyperparameter tuning

Feature engineering expanded the dataset from 9 to 12 structured features, including interaction terms such as   `seniority_company_interaction`, `technical_us_interaction`, and `desc_length_category_interaction`. Although stacking did not improve results (**accuracy dropped to 43.48%**), enhanced XGBoost improved performance to 61.26% accuracy.

Further optimization introduced **TF-IDF** features (100 terms) and additional keyword indicators, increasing the feature space to **17 total features**. Hyperparameter tuning via randomized search optimized XGBoost parameters (e.g., max_depth=8, n_estimators=300, learning_rate=0.2), raising performance to **64.07% accuracy**.

A final lightweight **Voting Ensemble** combining optimized models achieved the best performance at **65.04% accuracy**, representing a **+470% improvement** over baseline and an **+8% improvement** over the original XGBoost model.

### 5.1 Model Evaluation & Selection
Model performance was evaluated using Accuracy, Precision, Recall, F1-score, and 5-fold cross-validation to ensure generalization. The final comparison showed:

- Dummy Baseline: 11.37%
- Random Forest: 58.87%
- Logistic Regression: 54.49%
- XGBoost (Original): 60.04%
- XGBoost (Optimized): 64.07%
- **Voting Ensemble: 65.04% (Best)**

Cross-validation for the optimized XGBoost model yielded a mean accuracy of 63.93% (±0.0075), indicating stable performance across folds.

Feature importance analysis revealed that the most influential predictors were:

- `seniority_level`
- `num_categories`
- `has_technical_category`
- Description length features (`desc_word_count`, `desc_char_count`)
- Geographic indicator (`is_us`)
- Company size and interaction features

The Voting Ensemble was selected as the final model due to its highest overall accuracy and balanced precision-recall tradeoff. The final model:
- Uses**17 engineered features**
- Predicts across **23 job categories**
- Trained on **7,845 samples**
- Achieved **65.04% test accuracy**

## 5. Deployment
The final job category predictor was embedded directly into a user-friendly Streamlit application so that it can be used in real-world recruitment and HR workflows. The goal of deployment is to make job classification practical for recruiters, hiring managers, and job seekers, not just technically accurate.

The application is designed so that the user only needs to provide the core **job details: Job Title, Job Description, and optionally key Skills and Years of Experience**. All other calculations, such as category scoring, seniority estimation, and confidence calibration, are automatically handled in the background by the system. This ensures that the app remains intuitive and accessible while still leveraging the complex ensemble modeling and keyword-based logic developed during training.

This design choice was intentional because requiring users to manually input detailed probabilities, category mappings, or model internals would make the system cumbersome and reduce adoption. By keeping the input simple, the tool becomes practical and ready for real-world HR and recruiting use.

The deployed application provides not only primary job category predictions but also a confidence ranking for the top 5 categories, a seniority level estimation, and analytics on category distribution to give users actionable insights.

The deployed application can be accessed here:
After cloning the repository run this code in your terminal `streamlit run job_predictor_app.py`




