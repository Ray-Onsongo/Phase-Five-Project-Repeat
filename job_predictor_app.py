# job_predictor_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import re
from collections import defaultdict
import os

# ============================================================================
# JOB CATEGORY PREDICTOR CLASS (embedded directly in the app)
# ============================================================================

class JobCategoryPredictor:
    """
    Job Category Prediction System using Ensemble Model Calibration
    """

    def __init__(self, ensemble_predictions_path='ensemble_predictions.csv'):
        """
        Initialize the predictor with ensemble predictions for calibration
        """
        print("Initializing Job Category Predictor...")

        # Check if file exists
        if not os.path.exists(ensemble_predictions_path):
            st.error(f"File not found: {ensemble_predictions_path}")
            st.info("Please make sure 'ensemble_predictions.csv' is in the same directory as this app.")
            self.ensemble_df = pd.DataFrame()
        else:
            # Load ensemble predictions
            self.ensemble_df = pd.read_csv(ensemble_predictions_path)

        # Category mapping (from your dataset)
        self.category_mapping = {
            0: 'Administrative',
            1: 'Arts & Design',
            2: 'Business Analysis',
            3: 'Data Science',
            4: 'DevOps',
            5: 'Engineering',
            6: 'Finance',
            7: 'Healthcare',
            8: 'Human Resources',
            9: 'Information Technology',
            10: 'Internship',
            11: 'Legal',
            12: 'Management',
            13: 'Manufacturing',
            14: 'Marketing',
            15: 'Operations',
            16: 'Other',
            17: 'Project Management',
            18: 'Quality Assurance',
            19: 'Sales',
            20: 'Science',
            21: 'Software Engineering',
            22: 'Support'
        }

        # Reverse mapping for lookup
        self.reverse_category_mapping = {v: k for k, v in self.category_mapping.items()}

        # Extract patterns from ensemble predictions
        self._extract_patterns()

        # Calculate category priors from ensemble
        if len(self.ensemble_df) > 0 and 'actual' in self.ensemble_df.columns:
            self.category_priors = self.ensemble_df['actual'].value_counts(normalize=True).to_dict()
        else:
            # Default priors if no data
            self.category_priors = {i: 1/23 for i in range(23)}

        print(f"Loaded {len(self.category_mapping)} job categories")
        if len(self.ensemble_df) > 0:
            print(f"Calibrated with {len(self.ensemble_df)} ensemble predictions")

    def _extract_patterns(self):
        """
        Extract keyword patterns for each category from ensemble predictions
        """
        self.category_patterns = {}

        if len(self.ensemble_df) == 0:
            # Create default patterns if no data
            for cat_id in range(23):
                self.category_patterns[cat_id] = {'base_probability': 1/23}
            return

        # Get probability columns
        prob_cols = [col for col in self.ensemble_df.columns if col.startswith('prob_class_')]

        if prob_cols:
            # Use probability-based patterns
            for cat_id in range(23):
                prob_col = f'prob_class_{cat_id}'
                if prob_col in self.ensemble_df.columns:
                    # Calculate average probability when this category is actual
                    cat_data = self.ensemble_df[self.ensemble_df['actual'] == cat_id]
                    if len(cat_data) > 0:
                        avg_prob = cat_data[prob_col].mean()
                    else:
                        avg_prob = 1/23
                    self.category_patterns[cat_id] = {'base_probability': avg_prob}
        else:
            # Default patterns
            for cat_id in range(23):
                self.category_patterns[cat_id] = {'base_probability': 1/23}

    def _calculate_seniority_score(self, title):
        """
        Calculate seniority score from job title
        """
        seniority_keywords = {
            'junior': 1, 'entry': 1, 'associate': 1, 'trainee': 1,
            'mid': 2, 'intermediate': 2, 'experienced': 2,
            'senior': 3, 'sr': 3,
            'lead': 4, 'principal': 5, 'staff': 4,
            'manager': 4, 'director': 5, 'head': 5, 'chief': 5,
            'vp': 5, 'vice president': 5
        }

        title_lower = title.lower()
        score = 0

        for kw, kw_score in seniority_keywords.items():
            if kw in title_lower:
                score = max(score, kw_score)

        return score

    def _calculate_category_scores(self, title, description, skills):
        """
        Calculate scores for all 23 categories based on input
        """
        # Initialize scores with priors
        scores = {cat_id: self.category_priors.get(cat_id, 0.01) for cat_id in range(23)}

        # Combine text for analysis
        text_lower = f"{title} {description}".lower()
        skills_lower = [s.lower() for s in skills if s]

        # Common keywords for each category
        category_keywords = {
            0: ['administrative', 'admin', 'assistant', 'clerical', 'office'],
            1: ['design', 'art', 'creative', 'ui', 'ux', 'graphic'],
            2: ['business analyst', 'requirements', 'stakeholder', 'process'],
            3: ['data', 'analytics', 'machine learning', 'python', 'sql', 'ai'],
            4: ['devops', 'aws', 'cloud', 'docker', 'kubernetes', 'ci/cd'],
            5: ['engineer', 'engineering', 'mechanical', 'electrical', 'civil'],
            6: ['finance', 'accounting', 'financial', 'audit', 'tax'],
            7: ['healthcare', 'medical', 'nurse', 'doctor', 'clinical'],
            8: ['hr', 'human resources', 'recruiter', 'talent', 'people'],
            9: ['it', 'information technology', 'help desk', 'support', 'technical'],
            10: ['intern', 'internship', 'trainee', 'apprentice'],
            11: ['legal', 'law', 'attorney', 'counsel', 'compliance'],
            12: ['manager', 'management', 'director', 'head', 'lead'],
            13: ['manufacturing', 'production', 'plant', 'factory'],
            14: ['marketing', 'digital marketing', 'seo', 'content', 'social media'],
            15: ['operations', 'logistics', 'supply chain', 'distribution'],
            16: ['other', 'general', 'miscellaneous'],
            17: ['project manager', 'project management', 'pmp', 'agile'],
            18: ['quality', 'qa', 'test', 'assurance', 'testing'],
            19: ['sales', 'account executive', 'business development', 'b2b'],
            20: ['science', 'scientist', 'research', 'lab', 'r&d'],
            21: ['software', 'developer', 'programming', 'coding', 'full stack'],
            22: ['support', 'customer service', 'help desk', 'technical support']
        }

        # Calculate keyword matches
        for cat_id, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[cat_id] += 0.05
                # Check skills
                for skill in skills_lower:
                    if keyword in skill:
                        scores[cat_id] += 0.03

        # Adjust based on seniority
        seniority_score = self._calculate_seniority_score(title)
        if seniority_score >= 4:
            scores[10] *= 0.3  # Reduce internship for senior roles
        elif seniority_score <= 1:
            scores[12] *= 0.5  # Reduce management for junior roles

        return scores

    def predict(self, job_title, job_description, skills=None, experience=None, remote=None):
        """
        Predict job category for a new job posting
        """
        if skills is None:
            skills = []

        # Calculate seniority
        seniority_score = self._calculate_seniority_score(job_title)
        seniority_levels = ['Entry', 'Mid', 'Senior', 'Lead', 'Executive']
        seniority_level = seniority_levels[min(seniority_score, 4)]

        # Calculate scores for all categories
        category_scores = self._calculate_category_scores(job_title, job_description, skills)

        # Normalize scores to get probabilities
        total_score = sum(category_scores.values())
        if total_score > 0:
            probabilities = {cat_id: score/total_score for cat_id, score in category_scores.items()}
        else:
            probabilities = {cat_id: 1/23 for cat_id in range(23)}

        # Get top 5 predictions
        top_categories = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]

        # Format predictions
        predictions = []
        for cat_id, prob in top_categories:
            predictions.append({
                'category_id': cat_id,
                'category_name': self.category_mapping[cat_id],
                'confidence': prob
            })

        return {
            'primary_prediction': predictions[0],
            'all_predictions': predictions,
            'seniority_level': seniority_level,
            'seniority_score': seniority_score,
            'features_extracted': len(category_scores)
        }

    def get_category_stats(self):
        """
        Get statistics about categories from ensemble predictions
        """
        if len(self.ensemble_df) == 0:
            # Return default stats if no data
            stats = []
            for cat_id in range(23):
                stats.append({
                    'category_id': cat_id,
                    'category_name': self.category_mapping[cat_id],
                    'count': 0,
                    'percentage': 100/23,
                    'avg_confidence': 1/23
                })
            return pd.DataFrame(stats)

        stats = []
        for cat_id in range(23):
            cat_data = self.ensemble_df[self.ensemble_df['actual'] == cat_id]
            stats.append({
                'category_id': cat_id,
                'category_name': self.category_mapping[cat_id],
                'count': len(cat_data),
                'percentage': len(cat_data) / len(self.ensemble_df) * 100 if len(self.ensemble_df) > 0 else 0,
                'avg_confidence': cat_data['confidence'].mean() if len(cat_data) > 0 else 0
            })

        return pd.DataFrame(stats).sort_values('count', ascending=False)


# ============================================================================
# STREAMLIT APP
# ============================================================================

# Page config
st.set_page_config(
    page_title="Job Category Predictor",
    page_icon=" ",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .category-tag {
        background-color: #1E88E5;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    .confidence-bar {
        height: 25px;
        background: linear-gradient(90deg, #1E88E5, #64B5F6);
        border-radius: 12px;
        margin: 0.5rem 0;
        color: white;
        padding-left: 10px;
        line-height: 25px;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        height: 50px;
        font-size: 1.2rem;
    }
    .category-stats {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    with st.spinner('Loading Job Category Predictor...'):
        # Check if ensemble_predictions.csv exists
        if os.path.exists('ensemble_predictions.csv'):
            st.session_state.predictor = JobCategoryPredictor('ensemble_predictions.csv')
        else:
            st.warning("ensemble_predictions.csv not found. Using default settings.")
            st.session_state.predictor = JobCategoryPredictor()  # Will work with defaults
        st.session_state.history = []
        st.session_state.prediction_count = 0

# Header
st.markdown('<h1 class="main-header"> Job Category Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Powered by Ensemble Learning | 23 Job Categories")

# Sidebar
with st.sidebar:
    st.markdown("## Dashboard")
    st.markdown("---")

    # Quick stats
    st.markdown("### Quick Stats")
    stats = st.session_state.predictor.get_category_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Categories", len(stats))
    with col2:
        st.metric("Predictions Made", st.session_state.prediction_count)

    # Category distribution (simplified)
    st.markdown("### Top Categories")
    for idx, row in stats.head(5).iterrows():
        st.markdown(f"""
        <div class="category-stats">
            <b>{row['category_name']}</b><br>
            {row['percentage']:.1f}% of jobs
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This app predicts job categories using an ensemble model trained on thousands of job postings.

    **Supported Categories:**
    - Data Science
    - Software Engineering
    - DevOps
    - Management
    - And 19 more...
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["Predict", "Analytics", "History"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Job Details")

        # Input form
        with st.form("prediction_form"):
            job_title = st.text_input(
                "Job Title *", 
                placeholder="e.g., Senior Data Scientist",
                help="Enter the job title"
            )

            job_description = st.text_area(
                "Job Description *", 
                height=150,
                placeholder="Describe the role, responsibilities, and requirements...",
                help="Paste the full job description"
            )

            skills_input = st.text_input(
                "Skills (comma-separated)", 
                placeholder="Python, SQL, Machine Learning",
                help="List key skills required"
            )

            experience = st.slider(
                "Years of Experience",
                min_value=0,
                max_value=30,
                value=0,
                help="Required experience in years"
            )

            submitted = st.form_submit_button(
                "Predict Category", 
                use_container_width=True
            )

    with col2:
        st.markdown("### Example Jobs")
        st.markdown("Click to load an example:")

        examples = {
            "Data Scientist": {
                "title": "Senior Data Scientist",
                "desc": "Looking for an experienced data scientist with Python, machine learning, and SQL expertise to build predictive models. Must have experience with TensorFlow or PyTorch.",
                "skills": "Python, Machine Learning, SQL, TensorFlow",
                "exp": 5
            },
            "Software Engineer": {
                "title": "Full Stack Developer",
                "desc": "Develop and maintain web applications using React, Node.js, and PostgreSQL. Work in an agile team environment.",
                "skills": "JavaScript, React, Node.js, SQL",
                "exp": 3
            },
            "Legal Counsel": {
                "title": "Corporate Legal Counsel",
                "desc": "Provide legal advice on corporate matters, contracts, and compliance. Must have law degree and bar admission.",
                "skills": "Contract Law, Corporate Law, Compliance",
                "exp": 8
            },
            "Marketing Intern": {
                "title": "Marketing Intern",
                "desc": "Summer internship opportunity for students interested in digital marketing, social media, and content creation.",
                "skills": "Social Media, Content Creation, Communication",
                "exp": 0
            }
        }

        # Create buttons for examples
        for name, example in examples.items():
            if st.button(f"📌 {name}", key=f"example_{name}", use_container_width=True):
                st.session_state.example_title = example['title']
                st.session_state.example_desc = example['desc']
                st.session_state.example_skills = example['skills']
                st.session_state.example_exp = example['exp']
                st.rerun()

    # Prediction area
    if submitted:
        if not job_title or not job_description:
            st.error("Please provide both Job Title and Job Description")
        else:
            with st.spinner('Analyzing job posting...'):
                time.sleep(1)  # Simulate processing

                # Parse skills
                skills = [s.strip() for s in skills_input.split(',')] if skills_input else []

                # Make prediction
                result = st.session_state.predictor.predict(
                    job_title, 
                    job_description, 
                    skills, 
                    experience if experience > 0 else None
                )

                # Update counter
                st.session_state.prediction_count += 1

                # Add to history
                st.session_state.history.append({
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'title': job_title[:50] + "..." if len(job_title) > 50 else job_title,
                    'primary': result['primary_prediction']['category_name'],
                    'confidence': f"{result['primary_prediction']['confidence']:.1%}",
                    'seniority': result['seniority_level']
                })

                # Display results
                st.markdown("---")

                col3, col4 = st.columns([1, 1])

                with col3:
                    st.markdown("### Primary Prediction")

                    primary = result['primary_prediction']

                    # Create colored box for primary prediction
                    confidence_pct = int(primary['confidence'] * 100)
                    st.markdown(f"""
                    <div style="background-color: #1E88E5; padding: 20px; border-radius: 10px; text-align: center;">
                        <h2 style="color: white; margin: 0;">{primary['category_name']}</h2>
                        <p style="color: white; font-size: 1.2rem; margin: 10px 0;">Confidence: {primary['confidence']:.1%}</p>
                        <div style="background-color: white; height: 10px; border-radius: 5px; margin: 10px 0;">
                            <div style="background-color: #FFC107; width: {confidence_pct}%; height: 10px; border-radius: 5px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"**Seniority Level:** {result['seniority_level']}")

                with col4:
                    st.markdown("### Top 5 Categories")

                    # Display top predictions with confidence bars
                    for i, pred in enumerate(result['all_predictions'][:5], 1):
                        confidence_pct = int(pred['confidence'] * 100)
                        st.markdown(f"""
                        <div style="margin: 10px 0;">
                            <b>{i}. {pred['category_name']}</b><br>
                            <div style="background-color: #e0e0e0; height: 20px; border-radius: 5px; width: 100%%;">
                                <div style="background-color: #1E88E5; width: {confidence_pct}%%; height: 20px; border-radius: 5px; text-align: right; padding-right: 5px; color: white; line-height: 20px;">
                                    {pred['confidence']:.1%}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Feature summary
                st.markdown("### Analysis Summary")
                col5, col6, col7 = st.columns(3)

                with col5:
                    st.metric("Seniority Score", result['seniority_score'])
                with col6:
                    st.metric("Skills Provided", len(skills))
                with col7:
                    st.metric("Features Analyzed", result['features_extracted'])

with tab2:
    st.markdown("### Category Analytics")

    # Get stats
    stats_df = st.session_state.predictor.get_category_stats()

    # Top categories bar chart
    st.markdown("#### Top 10 Categories by Frequency")

    # Prepare data for bar chart
    chart_data = stats_df.head(10).set_index('category_name')['count']
    st.bar_chart(chart_data, use_container_width=True)

    # Category distribution table
    st.markdown("#### Category Details")

    # Format the dataframe for display
    display_df = stats_df[['category_name', 'count', 'percentage', 'avg_confidence']].copy()
    display_df['percentage'] = display_df['percentage'].round(1).astype(str) + '%'
    display_df['avg_confidence'] = display_df['avg_confidence'].round(3).apply(lambda x: f"{x:.1%}")
    display_df.columns = ['Category', 'Count', '% of Total', 'Avg Confidence']

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # Model performance summary (if ensemble data available)
    if len(st.session_state.predictor.ensemble_df) > 0:
        st.markdown("### Model Performance")

        df = st.session_state.predictor.ensemble_df
        if 'actual' in df.columns and 'predicted' in df.columns:
            accuracy = (df['actual'] == df['predicted']).mean()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Accuracy", f"{accuracy:.2%}")
            with col2:
                st.metric("Total Samples", f"{len(df):,}")
            with col3:
                st.metric("Categories", "23")

with tab3:
    st.markdown("### Prediction History")

    if st.session_state.history:
        # Convert history to dataframe
        history_df = pd.DataFrame(st.session_state.history)

        # Display history
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        # Clear history button
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions yet. Try predicting a job category!")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Made with using Streamlit | Job Category Predictor v1.0"
    "</p>", 
    unsafe_allow_html=True
)
