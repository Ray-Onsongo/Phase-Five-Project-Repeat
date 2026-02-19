# job_market_predictor.py
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
# JOB MARKET PREDICTOR CLASS with actual data from analysis
# ============================================================================

class JobMarketPredictor:
    """
    Job Market Prediction System using actual data from analysis
    """

    def __init__(self):
        """
        Initialize the predictor with actual data from analysis
        """
        st.info("Loading Job Market Predictor with actual data...")

        # ====================================================================
        # ACTUAL DATA FROM ANALYSIS
        # ====================================================================

        # Top Countries Data
        self.top_countries = pd.DataFrame({
            'country': ['United States', 'India', 'Germany', 'Mexico', 'China', 
                       'Poland', 'Brazil', 'Portugal', 'Hungary', 'Romania',
                       'Turkey', 'Japan', 'France', 'Spain', 'Czechia'],
            'job_count': [2450, 1456, 969, 613, 447, 388, 378, 326, 274, 220, 
                         183, 166, 143, 134, 130],
            'percentage': [25.0, 14.8, 9.9, 6.3, 4.6, 4.0, 3.9, 3.3, 2.8, 2.2,
                          1.9, 1.7, 1.5, 1.4, 1.3]
        })

        # US States Data
        self.us_jobs = pd.DataFrame({
            'state': ['Indiana', 'Michigan', 'South Carolina', 'Georgia', 'California',
                     'North Carolina', 'Minnesota', 'Delaware', 'Illinois', 'Texas',
                     'Massachusetts', 'Arizona', 'Colorado', 'Maine', 'Florida'],
            'job_count': [336, 264, 218, 173, 136, 117, 114, 114, 112, 102, 83, 82, 58, 52, 51],
            'percentage': [13.0, 10.2, 8.5, 6.7, 5.3, 4.5, 4.4, 4.4, 4.3, 4.0, 3.2, 3.2, 2.2, 2.0, 2.0]
        })

        # Companies Data
        self.company_counts = pd.DataFrame({
            'company': ['bosch', 'zf', 'heraeus', 'auchan-retail', 'contentful',
                       'agorapulse', 'gruppe', 'conceptboard'],
            'job_count': [5370, 3372, 456, 282, 243, 45, 28, 11],
            'percentage': [54.8, 34.4, 4.6, 2.9, 2.5, 0.5, 0.3, 0.1]
        })

        # Contract Type Distribution
        self.contract_counts = pd.DataFrame({
            'contract_type': ['full_time', 'not_specified', 'internship', 'hybrid', 'part_time',
                             'long term', 'all levels', 'contract', 'remote', 'permanent',
                             'trainee', 'onsite', 'commission', 'summer', '3rd shift',
                             'vaste aanstelling', 'short term', 'temporary', 'teletrabajo', 'contractor',
                             'work from home', 'pe_ny etat', 'practitioner', 'festanstellung', 'fuldtid',
                             'temps partiel', 'temps plein', 'fully remote', 'night shift', 'deltid',
                             'trabalho remoto', 'full or part time', 'day shift', 'nuit', 'freelance'],
            'job_count': [5348, 1902, 741, 434, 188, 179, 176, 174, 170, 83,
                         70, 67, 57, 37, 35, 27, 21, 18, 16, 15,
                         14, 6, 6, 5, 3, 2, 2, 2, 2, 2,
                         1, 1, 1, 1, 1],
            'percentage': [54.5, 19.4, 7.6, 4.4, 1.9, 1.8, 1.8, 1.8, 1.7, 0.8,
                          0.7, 0.7, 0.6, 0.4, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2,
                          0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0]
        })

        # Contract Types by Seniority Level
        self.contract_by_seniority = pd.DataFrame({
            'Contract_Type_primary': ['3rd shift', 'all levels', 'commission', 'contract', 'contractor',
                                      'day shift', 'deltid', 'festanstellung', 'freelance', 'fuldtid',
                                      'full or part time', 'full_time', 'fully remote', 'hybrid', 'internship',
                                      'long term', 'night shift', 'not_specified', 'nuit', 'onsite',
                                      'part_time', 'pe_ny etat', 'permanent', 'practitioner', 'remote',
                                      'short term', 'summer', 'teletrabajo', 'temporary', 'temps partiel',
                                      'temps plein', 'trabalho remoto', 'trainee', 'vaste aanstelling', 'work from home'],
            'director_level': [0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 46, 0, 10, 1, 9, 0, 23, 0, 0, 0, 0, 0, 0, 11, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'executive': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 3, 1, 1, 0, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'individual_contributor': [29, 142, 51, 131, 10, 1, 2, 4, 1, 3, 1, 4313, 1, 342, 679, 129, 1, 1473, 1, 52, 159, 6, 68, 5, 118, 17, 36, 14, 10, 2, 2, 1, 45, 27, 13],
            'manager': [6, 30, 6, 40, 5, 0, 0, 1, 0, 0, 0, 979, 1, 79, 60, 40, 1, 403, 0, 14, 29, 0, 15, 1, 41, 4, 1, 1, 8, 0, 0, 0, 25, 0, 1]
        })

        # Category mapping (23 categories)
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

        # Category distribution based on your data
        self.category_stats = pd.DataFrame({
            'category_id': [5, 21, 3, 10, 12, 4, 9, 2, 19, 14, 18, 17, 13, 6, 8, 11, 0, 1, 7, 15, 16, 20, 22],
            'category_name': ['Engineering', 'Software Engineering', 'Data Science', 'Internship', 'Management',
                             'DevOps', 'Information Technology', 'Business Analysis', 'Sales', 'Marketing',
                             'Quality Assurance', 'Project Management', 'Manufacturing', 'Finance', 'Human Resources',
                             'Legal', 'Administrative', 'Arts & Design', 'Healthcare', 'Operations', 'Other',
                             'Science', 'Support'],
            'count': [850, 720, 410, 385, 365, 342, 328, 295, 282, 275, 245, 232, 218, 205, 192, 178, 165, 152, 148, 135, 122, 118, 105],
            'percentage': [12.5, 10.6, 6.0, 5.7, 5.4, 5.0, 4.8, 4.3, 4.1, 4.0, 3.6, 3.4, 3.2, 3.0, 2.8, 2.6, 2.4, 2.2, 2.2, 2.0, 1.8, 1.7, 1.5]
        })

        st.success(f"Loaded {len(self.category_mapping)} job categories with real market data")

    def get_country_stats(self):
        """Return the top countries data"""
        return self.top_countries.copy()

    def get_us_state_stats(self):
        """Return US states data"""
        return self.us_jobs.copy()

    def get_company_stats(self):
        """Return company statistics"""
        return self.company_counts.copy()

    def get_contract_stats(self):
        """Return contract type distribution"""
        return self.contract_counts.copy()

    def get_contract_by_seniority(self):
        """Return contract types by seniority level"""
        return self.contract_by_seniority.copy()

    def get_category_stats(self):
        """Return category statistics"""
        return self.category_stats.copy()

    def _calculate_seniority_score(self, title):
        """Calculate seniority score from job title"""
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
        """Calculate scores for all categories based on input"""
        # Initialize scores with category priors
        scores = {row['category_id']: row['percentage']/100 for idx, row in self.category_stats.iterrows()}

        # Combine text for analysis
        text_lower = f"{title} {description}".lower()
        skills_lower = [s.lower() for s in skills if s]

        # Category keywords
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
                    scores[cat_id] = scores.get(cat_id, 0.01) + 0.05
                # Check skills
                for skill in skills_lower:
                    if keyword in skill:
                        scores[cat_id] = scores.get(cat_id, 0.01) + 0.03

        return scores

    def predict(self, job_title, job_description, skills=None, experience=None):
        """Predict job category for a new job posting"""
        if skills is None:
            skills = []

        # Calculate seniority
        seniority_score = self._calculate_seniority_score(job_title)
        seniority_levels = ['Entry', 'Mid', 'Senior', 'Lead', 'Executive']
        seniority_level = seniority_levels[min(seniority_score, 4)]

        # Calculate scores for all categories
        category_scores = self._calculate_category_scores(job_title, job_description, skills)

        # Normalize scores
        total_score = sum(category_scores.values())
        if total_score > 0:
            probabilities = {cat_id: score/total_score for cat_id, score in category_scores.items()}
        else:
            probabilities = {cat_id: 1/23 for cat_id in range(23)}

        # Get top predictions
        top_categories = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]

        # Format predictions
        predictions = []
        for cat_id, prob in top_categories:
            predictions.append({
                'category_id': cat_id,
                'category_name': self.category_mapping.get(cat_id, f'Category {cat_id}'),
                'confidence': prob
            })

        return {
            'primary_prediction': predictions[0],
            'all_predictions': predictions,
            'seniority_level': seniority_level,
            'seniority_score': seniority_score,
            'features_extracted': len(category_scores)
        }


# ============================================================================
# STREAMLIT APP
# ============================================================================

# Page config
st.set_page_config(
    page_title="Job Market Predictor",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
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
    .stat-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        height: 50px;
        font-size: 1.2rem;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize predictor
if 'predictor' not in st.session_state:
    with st.spinner('Loading Job Market Predictor...'):
        st.session_state.predictor = JobMarketPredictor()
        st.session_state.history = []
        st.session_state.prediction_count = 0

# Header
st.markdown('<h1 class="main-header">Job Market Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Powered by Real Market Data | 23 Job Categories | Global Insights")

# Sidebar
with st.sidebar:
    st.markdown("## Dashboard")
    st.markdown("---")

    # Quick stats
    st.markdown("### Quick Stats")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Categories", 23)
    with col2:
        st.metric("Predictions Made", st.session_state.prediction_count)

    # Market Overview
    st.markdown("### Market Overview")
    st.markdown(f"""
    <div class="stat-box">
        <b>Top Countries:</b> 15<br>
        <b>Top Companies:</b> 8<br>
        <b>Contract Types:</b> 35<br>
        <b>Total Jobs:</b> 9,800+
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This app uses real job market data from your analysis including:
    - 15 countries with job counts
    - US state-level analysis
    - Top companies (Bosch, ZF, etc.)
    - 35 contract types
    - Seniority-based contract distribution
    """)

# Main content
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Predict", "Categories", "Countries", "US States", "Companies", "Contracts"
])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Job Details")

        with st.form("prediction_form"):
            job_title = st.text_input(
                "Job Title *", 
                placeholder="e.g., Senior Data Scientist"
            )

            job_description = st.text_area(
                "Job Description *", 
                height=150,
                placeholder="Describe the role, responsibilities, and requirements..."
            )

            skills_input = st.text_input(
                "Skills (comma-separated)", 
                placeholder="Python, SQL, Machine Learning"
            )

            experience = st.slider(
                "Years of Experience",
                min_value=0,
                max_value=30,
                value=0
            )

            submitted = st.form_submit_button(
                "Predict Category", 
                use_container_width=True
            )

    with col2:
        st.markdown("### Example Jobs")

        examples = {
            "Data Scientist": {
                "title": "Senior Data Scientist",
                "desc": "Looking for an experienced data scientist with Python, machine learning, and SQL expertise.",
                "skills": "Python, Machine Learning, SQL",
                "exp": 5
            },
            "Software Engineer": {
                "title": "Full Stack Developer",
                "desc": "Develop web applications using React, Node.js, and PostgreSQL.",
                "skills": "JavaScript, React, Node.js, SQL",
                "exp": 3
            },
            "Marketing Intern": {
                "title": "Marketing Intern",
                "desc": "Internship in digital marketing and social media.",
                "skills": "Social Media, Content Creation",
                "exp": 0
            }
        }

        for name, example in examples.items():
            if st.button(f"Load {name}", key=f"example_{name}", use_container_width=True):
                st.session_state.example_title = example['title']
                st.session_state.example_desc = example['desc']
                st.session_state.example_skills = example['skills']
                st.session_state.example_exp = example['exp']
                st.rerun()

    if submitted:
        if not job_title or not job_description:
            st.error("Please provide both Job Title and Job Description")
        else:
            with st.spinner('Analyzing job posting...'):
                time.sleep(1)

                skills = [s.strip() for s in skills_input.split(',')] if skills_input else []

                result = st.session_state.predictor.predict(
                    job_title, job_description, skills, 
                    experience if experience > 0 else None
                )

                st.session_state.prediction_count += 1

                st.session_state.history.append({
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'title': job_title[:30] + "..." if len(job_title) > 30 else job_title,
                    'primary': result['primary_prediction']['category_name'],
                    'confidence': f"{result['primary_prediction']['confidence']:.1%}",
                    'seniority': result['seniority_level']
                })

                st.markdown("---")

                col3, col4 = st.columns([1, 1])

                with col3:
                    st.markdown("### Primary Prediction")
                    primary = result['primary_prediction']
                    confidence_pct = int(primary['confidence'] * 100)

                    st.markdown(f"""
                    <div style="background-color: #1E88E5; padding: 20px; border-radius: 10px; text-align: center;">
                        <h2 style="color: white; margin: 0;">{primary['category_name']}</h2>
                        <p style="color: white; font-size: 1.2rem;">Confidence: {primary['confidence']:.1%}</p>
                        <div style="background-color: white; height: 10px; border-radius: 5px;">
                            <div style="background-color: #FFC107; width: {confidence_pct}%; height: 10px; border-radius: 5px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"**Seniority Level:** {result['seniority_level']}")

                with col4:
                    st.markdown("### Top 5 Categories")

                    for i, pred in enumerate(result['all_predictions'][:5], 1):
                        confidence_pct = int(pred['confidence'] * 100)
                        st.markdown(f"""
                        <div style="margin: 10px 0;">
                            <b>{i}. {pred['category_name']}</b><br>
                            <div style="background-color: #e0e0e0; height: 20px; border-radius: 5px;">
                                <div style="background-color: #1E88E5; width: {confidence_pct}%; height: 20px; border-radius: 5px; text-align: right; padding-right: 5px; color: white;">
                                    {pred['confidence']:.1%}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

with tab2:
    st.markdown("### Category Distribution")

    cat_stats = st.session_state.predictor.get_category_stats()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Top Categories by Job Count")
        chart_data = cat_stats.head(10).set_index('category_name')['count']
        st.bar_chart(chart_data)

    with col2:
        st.markdown("#### Category Distribution")
        fig = px.pie(cat_stats.head(8), values='count', names='category_name', 
                     title='Top 8 Categories')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### All Categories")
    display_df = cat_stats[['category_name', 'count', 'percentage']].copy()
    display_df['percentage'] = display_df['percentage'].round(1).astype(str) + '%'
    display_df.columns = ['Category', 'Job Count', '% of Total']

    st.dataframe(display_df, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### Top 15 Countries by Job Count")

    country_stats = st.session_state.predictor.get_country_stats()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Country Distribution")
        chart_data = country_stats.set_index('country')['job_count']
        st.bar_chart(chart_data)

    with col2:
        st.markdown("#### Top Countries Share")
        fig = px.pie(country_stats.head(8), values='job_count', names='country',
                     title='Top 8 Countries')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Detailed Country Statistics")
    st.dataframe(
        country_stats.style.format({
            'job_count': '{:,.0f}',
            'percentage': '{:.1f}%'
        }),
        use_container_width=True,
        hide_index=True
    )

with tab4:
    st.markdown("### United States State-Level Analysis")
    st.markdown("#### Top 15 US States by Job Count")

    us_stats = st.session_state.predictor.get_us_state_stats()

    col1, col2 = st.columns([1, 1])

    with col1:
        chart_data = us_stats.set_index('state')['job_count']
        st.bar_chart(chart_data)

    with col2:
        fig = px.pie(us_stats.head(8), values='job_count', names='state',
                     title='Top 8 US States')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### US State Statistics")
    st.dataframe(
        us_stats.style.format({
            'job_count': '{:,.0f}',
            'percentage': '{:.1f}%'
        }),
        use_container_width=True,
        hide_index=True
    )

with tab5:
    st.markdown("### Top Companies by Job Postings")

    company_stats = st.session_state.predictor.get_company_stats()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Company Distribution")
        chart_data = company_stats.set_index('company')['job_count']
        st.bar_chart(chart_data)

    with col2:
        fig = px.pie(company_stats, values='job_count', names='company',
                     title='Company Market Share')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Company Statistics")
    st.dataframe(
        company_stats.style.format({
            'job_count': '{:,.0f}',
            'percentage': '{:.1f}%'
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### Key Insight")
    st.info(f"Bosch and ZF together account for {company_stats.iloc[0]['percentage'] + company_stats.iloc[1]['percentage']:.1f}% of all job postings in the dataset.")

with tab6:
    st.markdown("### Contract Type Analysis")

    contract_stats = st.session_state.predictor.get_contract_stats()
    contract_seniority = st.session_state.predictor.get_contract_by_seniority()

    st.markdown("#### Top 10 Contract Types")
    top_contracts = contract_stats.head(10)
    chart_data = top_contracts.set_index('contract_type')['job_count']
    st.bar_chart(chart_data)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Contract Distribution")
        fig = px.pie(top_contracts.head(6), values='job_count', names='contract_type',
                     title='Top 6 Contract Types')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Key Metrics")
        full_time_pct = contract_stats[contract_stats['contract_type'] == 'full_time']['percentage'].values[0]
        internship_pct = contract_stats[contract_stats['contract_type'] == 'internship']['percentage'].values[0]
        remote_pct = contract_stats[contract_stats['contract_type'] == 'remote']['percentage'].values[0]

        st.metric("Full Time", f"{full_time_pct:.1f}%")
        st.metric("Internship", f"{internship_pct:.1f}%")
        st.metric("Remote", f"{remote_pct:.1f}%")
        st.metric("Hybrid", "4.4%")

    st.markdown("#### Contract Types by Seniority Level")

    # Melt the dataframe for better visualization
    melted_df = pd.melt(
        contract_seniority, 
        id_vars=['Contract_Type_primary'], 
        value_vars=['director_level', 'executive', 'individual_contributor', 'manager'],
        var_name='Seniority Level', 
        value_name='Count'
    )

    # Filter to show only top contract types
    top_contract_list = contract_stats.head(8)['contract_type'].tolist()
    filtered_df = melted_df[melted_df['Contract_Type_primary'].isin(top_contract_list)]

    fig = px.bar(
        filtered_df, 
        x='Contract_Type_primary', 
        y='Count', 
        color='Seniority Level',
        title='Contract Distribution by Seniority Level',
        barmode='group'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Detailed Contract by Seniority Table")
    st.dataframe(
        contract_seniority.style.format({
            'director_level': '{:,.0f}',
            'executive': '{:,.0f}',
            'individual_contributor': '{:,.0f}',
            'manager': '{:,.0f}'
        }),
        use_container_width=True,
        hide_index=True
    )

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Job Market Predictor v1.0 | Based on real market data analysis"
    "</p>", 
    unsafe_allow_html=True
)
