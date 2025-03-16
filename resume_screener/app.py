import os
import streamlit as st
import pandas as pd
import tempfile
from typing import Dict, Any, List
import time
import logging
from dotenv import load_dotenv

from utils.pdf_parser import PDFParser
from services.ats_scoring_service import ATSScoringService
from services.job_market_service import JobMarketService
from utils.visualization import VisualizationHelper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize services
ats_service = ATSScoringService()
job_market_service = JobMarketService()

# Set page configuration
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("📑 AI-Powered Resume Screening Tool")
st.markdown("""
This tool helps recruiters analyze job descriptions, score resumes against requirements, 
detect AI-generated content, and rank candidates automatically.
""")

# Session state initialization
if 'job_requirements' not in st.session_state:
    st.session_state.job_requirements = None
if 'processed_resumes' not in st.session_state:
    st.session_state.processed_resumes = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'job_description' not in st.session_state:
    st.session_state.job_description = ""
if 'market_insights' not in st.session_state:
    st.session_state.market_insights = None

# Create sidebar for input options
st.sidebar.header("Setup")

# Job description input
st.sidebar.subheader("1. Job Description")
job_desc_option = st.sidebar.radio("How would you like to input the job description?", 
                                   ["Text Input", "Upload File"])

if job_desc_option == "Text Input":
    job_description = st.sidebar.text_area(
        "Paste job description here", 
        value=st.session_state.job_description, 
        height=300
    )
    if job_description != st.session_state.job_description:
        st.session_state.job_description = job_description
        st.session_state.job_requirements = None  # Reset analysis if description changes
else:
    job_desc_file = st.sidebar.file_uploader("Upload job description", type=["txt", "pdf"])
    if job_desc_file:
        if job_desc_file.type == "application/pdf":
            # Save PDF to temp file and extract text
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(job_desc_file.getvalue())
                temp_path = temp_file.name
                
            job_description = PDFParser.extract_resume_text(temp_path)
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        else:
            # Read text file
            job_description = job_desc_file.getvalue().decode("utf-8")
            
        st.session_state.job_description = job_description
        st.session_state.job_requirements = None  # Reset analysis if description changes

# Resume upload
st.sidebar.subheader("2. Candidate Resumes")
resume_files = st.sidebar.file_uploader("Upload resumes (PDF)", 
                                       type=["pdf"], 
                                       accept_multiple_files=True)

# Analysis button
analyze_button = st.sidebar.button("Analyze Job & Resumes", use_container_width=True)

# Job market insights toggle
if st.sidebar.checkbox("Include job market insights"):
    st.session_state.include_market_insights = True
else:
    st.session_state.include_market_insights = False

# Main content area
tab1, tab2, tab3 = st.tabs(["Job Analysis", "Candidate Ranking", "Detailed View"])

# Tab 1: Job Analysis
with tab1:
    if analyze_button or st.session_state.job_requirements:
        with st.spinner("Analyzing job description..."):
            # Only re-analyze if needed
            if analyze_button or not st.session_state.job_requirements:
                if st.session_state.job_description:
                    st.session_state.job_requirements = ats_service.analyze_job_description(st.session_state.job_description)
                    
                    # Get job market insights if enabled
                    if st.session_state.include_market_insights and st.session_state.job_requirements:
                        job_title = st.session_state.job_requirements.get("job_title", "")
                        required_skills = st.session_state.job_requirements.get("required_skills", [])
                        
                        # Make sure we have valid skills data
                        if required_skills and isinstance(required_skills, list) and required_skills[0] != "None specified":
                            st.session_state.market_insights = job_market_service.get_job_market_comparison(
                                job_title=job_title,
                                required_skills=required_skills
                            )
                        else:
                            # Get general market insights without skill analysis
                            st.session_state.market_insights = job_market_service.get_job_market_comparison(
                                job_title=job_title
                            )
            
            # Display job analysis results
            if st.session_state.job_requirements:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Job Requirements")
                    st.markdown(f"**Title:** {st.session_state.job_requirements.get('job_title', 'Unknown')}")
                    st.markdown(f"**Industry:** {st.session_state.job_requirements.get('industry', 'Unknown')}")
                    st.markdown(f"**Level:** {st.session_state.job_requirements.get('seniority_level', 'Not specified')}")
                    st.markdown(f"**Experience:** {st.session_state.job_requirements.get('years_of_experience', 'Not specified')}")
                    st.markdown(f"**Education:** {st.session_state.job_requirements.get('education_requirements', 'Not specified')}")
                    
                    # Required skills
                    st.subheader("Required Skills")
                    required_skills = st.session_state.job_requirements.get("required_skills", [])
                    
                    # Make sure we have valid skills
                    if required_skills and isinstance(required_skills, list) and required_skills[0] != "None specified":
                        # Display as tags
                        html_skills = ""
                        for skill in required_skills:
                            html_skills += f"""<span style="background-color: #0078ff; color: white; 
                                            padding: 0.2rem 0.5rem; margin: 0.2rem; border-radius: 1rem; 
                                            display: inline-block; font-size: 0.85rem;">{skill}</span>"""
                        st.markdown(html_skills, unsafe_allow_html=True)
                    else:
                        st.write("No required skills specified in the job description.")
                        
                    # Preferred skills
                    st.subheader("Preferred Skills")
                    preferred_skills = st.session_state.job_requirements.get("preferred_skills", [])
                    
                    # Make sure we have valid skills
                    if preferred_skills and isinstance(preferred_skills, list) and preferred_skills[0] != "None specified":
                        # Display as tags
                        html_skills = ""
                        for skill in preferred_skills:
                            html_skills += f"""<span style="background-color: #4CAF50; color: white; 
                                            padding: 0.2rem 0.5rem; margin: 0.2rem; border-radius: 1rem; 
                                            display: inline-block; font-size: 0.85rem;">{skill}</span>"""
                        st.markdown(html_skills, unsafe_allow_html=True)
                    else:
                        st.write("No preferred skills specified in the job description.")
                    
                with col2:
                    st.subheader("Job Market Insights")
                    if st.session_state.include_market_insights and st.session_state.market_insights:
                        market_insights = st.session_state.market_insights
                        st.markdown(market_insights.get("market_summary", "No market data available."))
                        
                        # Skills demand visualization
                        skills_demand = market_insights.get("skills_demand", {})
                        if skills_demand:
                            st.subheader("Skill Demand")
                            
                            # Prepare data for visualization
                            for skill, demand in skills_demand.items():
                                if demand.lower() == "high":
                                    color = "#1E88E5"  # Blue
                                elif demand.lower() == "medium":
                                    color = "#43A047"  # Green
                                elif demand.lower() == "low":
                                    color = "#E53935"  # Red
                                else:
                                    color = "#9E9E9E"  # Grey
                                    
                                st.markdown(
                                    f"""<div style="display: flex; align-items: center; margin-bottom: 8px;">
                                        <div style="width: 150px; overflow: hidden; text-overflow: ellipsis;">{skill}</div>
                                        <div style="background-color: {color}; width: 100px; height: 20px; 
                                                border-radius: 3px; color: white; text-align: center; 
                                                line-height: 20px; font-size: 0.8rem;">{demand.upper()}</div>
                                    </div>""", 
                                    unsafe_allow_html=True
                                )
                    else:
                        st.write("Enable job market insights to see data about skill demand and salary trends.")
                        st.markdown("*Tip: Use market insights to align your candidate requirements with current trends.*")
                
                # Key Responsibilities section
                st.subheader("Key Responsibilities")
                responsibilities = st.session_state.job_requirements.get("key_responsibilities", [])
                if responsibilities and isinstance(responsibilities, list) and responsibilities[0] != "None specified":
                    for resp in responsibilities:
                        st.markdown(f"• {resp}")
                else:
                    st.write("No key responsibilities specified in the job description.")
                    
                # Interview questions
                st.subheader("Suggested Interview Questions")
                interview_questions = ats_service.job_analyzer.generate_interview_questions(st.session_state.job_requirements)
                if interview_questions:
                    for i, question in enumerate(interview_questions, 1):
                        st.markdown(f"{i}. {question}")
            else:
                st.warning("Please enter a job description and click 'Analyze' to see results.")

# Tab 2: Candidate Ranking
with tab2:
    if analyze_button or (st.session_state.job_requirements and resume_files):
        with st.spinner("Processing resumes and analyzing candidates..."):
            # Process the resumes if needed
            if analyze_button or resume_files:
                if resume_files and st.session_state.job_requirements:
                    # Extract text from uploaded resumes
                    resume_texts = {}
                    for uploaded_file in resume_files:
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_path = temp_file.name
                            
                        # Extract text
                        pdf_result = PDFParser.parse_resume(temp_path)
                        if pdf_result["success"]:
                            # Use the enhanced structured data from our improved PDF parser
                            if "structured_data" in pdf_result:
                                resume_data = pdf_result["structured_data"]
                                resume_data["file_name"] = uploaded_file.name  # Add filename for reference
                            else:
                                # Fall back to just using the text if structured data not available
                                resume_text = pdf_result["text"]
                                resume_texts[uploaded_file.name] = resume_text
                        
                        # Clean up
                        try:
                            os.unlink(temp_path)
                        except Exception:
                            pass
                    
                    # Process everything
                    if resume_texts:
                        st.session_state.analysis_results = ats_service.process_job_and_resumes(
                            st.session_state.job_description,
                            resume_texts
                        )
                    
            # Display candidate ranking
            if st.session_state.analysis_results and "ranking_results" in st.session_state.analysis_results:
                ranking_results = st.session_state.analysis_results["ranking_results"]
                
                # Show insights
                st.subheader("Ranking Insights")
                st.markdown(ranking_results.get("insights", "No insights available."))
                
                # Display candidates sorted by rank
                st.subheader("Ranked Candidates")
                
                if "ranked_candidates" in ranking_results and ranking_results["ranked_candidates"]:
                    ranked_candidates = ranking_results["ranked_candidates"]
                    
                    # Create DataFrame for tabular view
                    candidate_data = []
                    for i, candidate in enumerate(ranked_candidates, 1):
                        candidate_data.append({
                            "Rank": i,
                            "Name": candidate.get("name", "Unknown"),
                            "ATS Score": f"{candidate.get('ats_score', 0):.1f}",
                            "Skills Match": f"{candidate.get('skill_match', 0):.1f}%",
                            "Experience Match": f"{candidate.get('experience_match', 0):.1f}%",
                            "Education Match": f"{candidate.get('education_match', 0):.1f}%",
                            "Authenticity": f"{candidate.get('authenticity_score', 0)*100:.1f}%",
                            "Composite Score": f"{candidate.get('composite_score', 0):.1f}",
                            "File": candidate.get("file_name", ""),
                        })
                    
                    df = pd.DataFrame(candidate_data)
                    
                    # Style the dataframe
                    st.dataframe(
                        df,
                        column_config={
                            "Rank": st.column_config.NumberColumn(format="%d"),
                            "ATS Score": st.column_config.ProgressColumn(
                                "ATS Score", format="%.1f", min_value=0, max_value=100),
                            "Authenticity": st.column_config.ProgressColumn(
                                "Authenticity", format="%.1f%%", min_value=0, max_value=100),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Visualizations
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        st.subheader("ATS Score Distribution")
                        ats_scores = [c.get("ats_score", 0) for c in ranked_candidates]
                        fig = VisualizationHelper.create_ats_score_histogram(ats_scores)
                        st.pyplot(fig)
                    
                    with viz_col2:
                        st.subheader("Candidate Comparison")
                        # Select top 5 candidates for comparison
                        top_candidates = ranked_candidates[:min(5, len(ranked_candidates))]
                        metrics = ["ats_score", "skill_match", "experience_match", "education_match", "authenticity_score"]
                        fig = VisualizationHelper.create_candidate_comparison_chart(top_candidates, metrics)
                        if fig:
                            st.pyplot(fig)
                        else:
                            st.info("Insufficient data for comparison visualization.")
                    
                    # AI Content Detection Summary
                    st.subheader("AI Content Detection")
                    auth_scores = [c.get("authenticity_score", 0.5) for c in ranked_candidates[:5]]
                    auth_fig = VisualizationHelper.create_ai_confidence_gauge(auth_scores)
                    if auth_fig:
                        st.pyplot(auth_fig)
                    
                else:
                    st.info("No candidates to rank. Please upload resumes to see results.")
            
            elif resume_files:
                st.warning("Processing candidates. Please wait...")
            else:
                st.info("Upload candidate resumes to see ranking results.")

# Tab 3: Detailed View
with tab3:
    if st.session_state.analysis_results and "candidates" in st.session_state.analysis_results:
        candidates = st.session_state.analysis_results["candidates"]
        job_requirements = st.session_state.analysis_results["job_requirements"]
        
        if candidates:
            # Candidate selector
            candidate_names = [f"{c.get('name', 'Unknown')} ({c.get('file_name', '')})" for c in candidates]
            selected_candidate = st.selectbox("Select candidate for detailed view", candidate_names)
            
            # Find selected candidate
            selected_idx = candidate_names.index(selected_candidate)
            candidate = candidates[selected_idx]
            
            if candidate:
                # Create tabs for different sections of detailed view
                detail_tab1, detail_tab2, detail_tab3 = st.tabs([
                    "Resume Overview", "ATS Scoring", "AI Content Analysis"
                ])
                
                # Tab 1: Resume Overview
                with detail_tab1:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.header(candidate.get("name", "Unknown Candidate"))
                        
                        # Contact info
                        contact_info = candidate.get("contact_info", {})
                        contact_html = f"""
                        <div style="margin-bottom: 20px;">
                            <p style="margin: 0;">{contact_info.get("email", "")}</p>
                            <p style="margin: 0;">{contact_info.get("phone", "")}</p>
                            <p style="margin: 0;">{contact_info.get("location", "")}</p>
                        </div>
                        """
                        st.markdown(contact_html, unsafe_allow_html=True)
                        
                        # Summary
                        if "summary" in candidate and candidate["summary"]:
                            st.subheader("Professional Summary")
                            st.markdown(candidate["summary"])
                            
                        # Experience
                        st.subheader("Experience")
                        experience = candidate.get("experience", [])
                        if experience:
                            for exp in experience:
                                exp_title = exp.get("title", "")
                                exp_company = exp.get("company", "")
                                exp_duration = exp.get("duration", "")
                                exp_dates = f"{exp.get('start_date', '')} - {exp.get('end_date', '')}"
                                exp_desc = exp.get("description", "")
                                
                                st.markdown(f"**{exp_title}** at **{exp_company}**")
                                st.markdown(f"*{exp_dates}* ({exp_duration})")
                                st.markdown(exp_desc)
                                st.markdown("---")
                        else:
                            st.write("No experience information available.")
                            
                        # Education
                        st.subheader("Education")
                        education = candidate.get("education", [])
                        if education:
                            for edu in education:
                                edu_degree = edu.get("degree", "")
                                edu_institution = edu.get("institution", "")
                                edu_field = edu.get("field", "")
                                edu_date = edu.get("graduation_date", "")
                                
                                st.markdown(f"**{edu_degree}** in *{edu_field}*")
                                st.markdown(f"{edu_institution}, {edu_date}")
                        else:
                            st.write("No education information available.")
                            
                    with col2:
                        # Skills
                        st.subheader("Skills")
                        skills = candidate.get("skills", [])
                        
                        if skills:
                            # Split into matching vs other skills
                            matching_skills = set(candidate.get("matching_skills", []))
                            
                            # Display as tags
                            html_skills = ""
                            for skill in skills:
                                if skill.lower() in (s.lower() for s in matching_skills):
                                    # Matching skill - highlight with job's color
                                    html_skills += f"""<span style="background-color: #0078ff; color: white; 
                                                padding: 0.2rem 0.5rem; margin: 0.2rem; border-radius: 1rem; 
                                                display: inline-block; font-size: 0.85rem;">{skill}</span>"""
                                else:
                                    # Other skill - neutral color
                                    html_skills += f"""<span style="background-color: #9e9e9e; color: white; 
                                                padding: 0.2rem 0.5rem; margin: 0.2rem; border-radius: 1rem; 
                                                display: inline-block; font-size: 0.85rem;">{skill}</span>"""
                            st.markdown(html_skills, unsafe_allow_html=True)
                        else:
                            st.write("No skills information available.")
                            
                        # Certifications
                        st.subheader("Certifications")
                        certifications = candidate.get("certifications", [])
                        if certifications:
                            for cert in certifications:
                                st.markdown(f"• {cert}")
                        else:
                            st.write("No certification information available.")
                            
                        # Languages
                        st.subheader("Languages")
                        languages = candidate.get("languages", [])
                        if languages:
                            for lang in languages:
                                st.markdown(f"• {lang}")
                        else:
                            st.write("No language information available.")
                
                # Tab 2: ATS Scoring
                with detail_tab2:
                    st.header("ATS Scoring Analysis")
                    
                    # Main scores
                    col1, col2, col3, col4 = st.columns(4)
                    
                    col1.metric("Overall ATS Score", f"{candidate.get('ats_score', 0):.1f}/100")
                    col2.metric("Skill Match", f"{candidate.get('skill_match', 0):.1f}%")
                    col3.metric("Experience Match", f"{candidate.get('experience_match', 0):.1f}%")
                    col4.metric("Education Match", f"{candidate.get('education_match', 0):.1f}%")
                    
                    st.markdown("---")
                    
                    # Skill matching details
                    st.subheader("Skill Matching")
                    
                    # Create two columns
                    match_col, missing_col = st.columns(2)
                    
                    with match_col:
                        st.markdown("#### Matching Skills")
                        matching_skills = candidate.get("matching_skills", [])
                        if matching_skills:
                            for skill in matching_skills:
                                st.markdown(f"✅ {skill}")
                        else:
                            st.write("No matching skills found.")
                    
                    with missing_col:
                        st.markdown("#### Missing Skills")
                        missing_skills = candidate.get("missing_skills", [])
                        if missing_skills:
                            for skill in missing_skills:
                                st.markdown(f"❌ {skill}")
                        else:
                            st.write("No missing skills!")
                    
                    st.markdown("---")
                    
                    # Detailed analysis
                    st.subheader("Detailed Analysis")
                    detailed_analysis = candidate.get("detailed_analysis", "No detailed analysis available.")
                    st.markdown(detailed_analysis)
                    
                    # Visualization
                    try:
                        # Convert matching_skills to dict for radar chart
                        candidate_skills = {skill: 100 for skill in candidate.get("matching_skills", [])}
                        required_skills = {skill: 100 for skill in job_requirements.get("required_skills", [])}
                        
                        # Only create visualization if there are skills
                        if candidate_skills and required_skills:
                            st.subheader("Skills Match Visualization")
                            radar_fig = VisualizationHelper.create_skill_match_radar(candidate_skills, required_skills)
                            st.pyplot(radar_fig)
                    except Exception as e:
                        st.error(f"Error generating skills visualization: {e}")
                
                # Tab 3: AI Content Analysis
                with detail_tab3:
                    st.header("AI-Generated Content Detection")
                    
                    # Overall authenticity score
                    authenticity_score = candidate.get("authenticity_score", 0.5)
                    
                    # Display score with appropriate color
                    if authenticity_score >= 0.8:
                        score_color = "green"
                        assessment = "Likely human-written"
                    elif authenticity_score >= 0.6:
                        score_color = "orange"
                        assessment = "Possibly contains some AI-generated content"
                    else:
                        score_color = "red"
                        assessment = "Likely contains significant AI-generated content"
                    
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                        <h3 style="margin-top: 0;">Authenticity Score: <span style="color: {score_color};">{authenticity_score:.2f}</span></h3>
                        <p><strong>Assessment:</strong> {assessment}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display analysis
                    st.subheader("AI Detection Analysis")
                    ai_analysis = candidate.get("ai_analysis", "No AI analysis available.")
                    st.markdown(ai_analysis)
                    
                    # Display flagged sections if any
                    flagged_sections = candidate.get("flagged_sections", {})
                    if flagged_sections:
                        st.subheader("Flagged Sections")
                        
                        for section, details in flagged_sections.items():
                            with st.expander(f"{section.title()} - Score: {details.get('score', 0):.2f}"):
                                st.markdown(f"**Why flagged:** {details.get('why', 'Unknown reason')}")
                                
                    else:
                        st.info("No sections were flagged for potential AI-generated content.")
                    
                    # Tips for reviewers
                    with st.expander("Tips for Evaluating AI Content"):
                        st.markdown("""
                        When reviewing potentially AI-generated content:
                        
                        1. **Ask follow-up questions** in interviews about specific resume details
                        2. **Request work samples** to verify skills claimed in the resume
                        3. **Look for inconsistencies** in the narrative or timeline
                        4. **Consider technical assessments** for crucial skills
                        5. **Focus on substance over style** - great writing doesn't always mean great skills
                        """)
        else:
            st.info("No candidates available for detailed view. Please upload resumes to see results.")
    else:
        st.info("Please upload and analyze resumes to see detailed candidate information.")
