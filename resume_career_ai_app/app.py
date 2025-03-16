import streamlit as st
import os
import tempfile
import json
import time
from datetime import datetime
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from crewai import Crew

# Import utility functions and agent classes
from utils.pdf_parser import PDFParser
from utils.serper_scraper import SerperScraper
from agents.resume_analyzer import ResumeAnalyzerAgent
from agents.job_market_analyzer import JobMarketAnalyzerAgent
from agents.resume_scorer import ResumeScoringAgent
from agents.career_guidance import CareerGuidanceAgent
from agents.chatbot_agent import ChatbotAgent

# Page configuration
st.set_page_config(
    page_title="AI Resume & Career Advisor",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = None
if 'market_data' not in st.session_state:
    st.session_state.market_data = None
if 'scoring_data' not in st.session_state:
    st.session_state.scoring_data = None
if 'career_guidance' not in st.session_state:
    st.session_state.career_guidance = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chatbot_agent' not in st.session_state:
    st.session_state.chatbot_agent = None

# Function to display the header
def display_header():
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://img.icons8.com/color/96/000000/resume.png", width=80)
    with col2:
        st.title("AI Resume & Career Advisor")
    st.markdown("""
    Upload your resume to get personalized career insights, skill gap analysis, and guidance.
    """)
    st.divider()

# Function to initialize API keys
def initialize_api_keys():
    # Load API keys directly from .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    serper_key = os.getenv("SERPER_API_KEY")
    
    # Check if keys are available
    api_keys_provided = bool(openrouter_key and serper_key)
    
    # Instead of showing inputs in the sidebar, just show status
    with st.sidebar:
        st.header("API Configuration")
        
        if api_keys_provided:
            st.success("✅ API keys loaded from environment")
        else:
            st.error("❌ API keys not found in environment")
            st.info(
                """
                Please add your API keys to the .env file:
                
                ```
                OPENROUTER_API_KEY=your_key_here
                SERPER_API_KEY=your_key_here
                ```
                """
            )
    
    return openrouter_key, serper_key, api_keys_provided

# Function to process the resume
def process_resume(file_path, openrouter_key, serper_key, career_goal=None):
    with st.spinner("Analyzing your resume..."):
        try:
            # Parse the PDF
            pdf_result = PDFParser.parse_resume(file_path)
            
            if not pdf_result["success"]:
                st.error(f"Error parsing resume: {pdf_result.get('error', 'Unknown error')}")
                return None, None, None, None
            
            resume_text = pdf_result["text"]
            
            # Initialize agents
            resume_analyzer = ResumeAnalyzerAgent(openrouter_key)
            
            # Extract resume information
            st.text("Extracting resume information...")
            resume_data = resume_analyzer.extract_resume_information(resume_text, career_goal)
            
            if "error" in resume_data:
                st.error(f"Error analyzing resume: {resume_data['error']}")
                return None, None, None, None
                
            st.text("Resume information extracted successfully.")
            
            # Initialize the job market analyzer with Serper
            serper_scraper = SerperScraper(serper_key)
            job_market_analyzer = JobMarketAnalyzerAgent(openrouter_key, serper_scraper)
            
            # Get market trends based on resume and career goal
            st.text("Analyzing job market trends...")
            market_data = job_market_analyzer.get_trending_skills_for_profile(resume_data, career_goal)
            
            if "error" in market_data:
                st.error(f"Error analyzing job market: {market_data['error']}")
                return resume_data, None, None, None
                
            st.text("Job market trends analyzed successfully.")
            
            # Score the resume with career goal in mind
            st.text("Scoring resume against market requirements...")
            resume_scorer = ResumeScoringAgent(openrouter_key)
            scoring_data = resume_scorer.score_resume(resume_data, market_data, career_goal)
            
            if "error" in scoring_data:
                st.error(f"Error scoring resume: {scoring_data['error']}")
                return resume_data, market_data, None, None
                
            st.text("Resume scored successfully.")
            
            # Generate career guidance with career goal in mind
            st.text("Generating career guidance...")
            career_guide = CareerGuidanceAgent(openrouter_key)
            guidance_data = career_guide.generate_career_recommendations(
                resume_data, market_data, scoring_data, career_goal
            )
            
            if "error" in guidance_data:
                st.error(f"Error generating career guidance: {guidance_data['error']}")
                return resume_data, market_data, scoring_data, None
                
            st.text("Career guidance generated successfully.")
            
            # Initialize the chatbot agent for future interactions
            st.session_state.chatbot_agent = ChatbotAgent(openrouter_key)
            
            return resume_data, market_data, scoring_data, guidance_data
            
        except Exception as e:
            st.error(f"Unexpected error during processing: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None, None, None, None

# Function to display resume analysis
def display_resume_analysis(resume_data):
    st.header("📄 Resume Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "personalInformation" in resume_data:
            personal_info = resume_data["personalInformation"]
            st.subheader("Personal Information")
            
            name = personal_info.get("name", "Not available")
            st.write(f"**Name:** {name}")
            
            if "email" in personal_info:
                st.write(f"**Email:** {personal_info['email']}")
                
            if "phone" in personal_info:
                st.write(f"**Phone:** {personal_info['phone']}")
                
            if "location" in personal_info:
                st.write(f"**Location:** {personal_info['location']}")
    
    with col2:
        if "skills" in resume_data and resume_data["skills"]:
            st.subheader("Skills")
            skills = resume_data["skills"]
            # Display as pills/tags
            skills_html = ""
            for skill in skills:
                skills_html += f'<span style="background-color: #e1f5fe; padding: 3px 10px; border-radius: 15px; margin-right: 5px; margin-bottom: 5px; display: inline-block; font-size: 0.8em;">{skill}</span>'
            st.markdown(skills_html, unsafe_allow_html=True)
    
    # Education
    if "education" in resume_data and resume_data["education"]:
        st.subheader("Education")
        for edu in resume_data["education"]:
            degree = edu.get("degree", "Degree")
            institution = edu.get("institution", "Institution")
            dates = f"{edu.get('startDate', '')} - {edu.get('endDate', '')}"
            st.markdown(f"**{degree}** - {institution} ({dates})")
            if "gpa" in edu and edu["gpa"]:
                st.markdown(f"GPA: {edu['gpa']}")
    
    # Work Experience
    if "workExperience" in resume_data and resume_data["workExperience"]:
        st.subheader("Work Experience")
        for exp in resume_data["workExperience"]:
            job_title = exp.get("jobTitle", "Position")
            company = exp.get("company", "Company")
            dates = f"{exp.get('startDate', '')} - {exp.get('endDate', '')}"
            st.markdown(f"**{job_title}** at {company} ({dates})")
            if "description" in exp:
                with st.expander("Details"):
                    st.write(exp["description"])
    
    # Projects (if available)
    if "projects" in resume_data and resume_data["projects"]:
        st.subheader("Projects")
        for project in resume_data["projects"]:
            project_name = project.get("name", "Project")
            st.markdown(f"**{project_name}**")
            if "description" in project:
                st.write(project["description"])
    
    # Certifications (if available)
    if "certifications" in resume_data and resume_data["certifications"]:
        st.subheader("Certifications")
        for cert in resume_data["certifications"]:
            cert_name = cert.get("name", "Certification")
            issuer = cert.get("issuer", "")
            date = cert.get("date", "")
            st.markdown(f"**{cert_name}** from {issuer} ({date})")

# Function to display job market trends
def display_market_trends(market_data):
    st.header("🌐 Job Market Trends")
    
    # Add debug option
    debug_view = st.checkbox("Show raw data (debug)", key="debug_market", value=True)  # Set to true for debugging
    if debug_view:
        st.json(market_data)
        
    col1, col2 = st.columns(2)
    
    # Technical Skills - handle different formats
    tech_skills = None
    
    # Try different possible keys - ADD YOUR SPECIFIC KEY
    for key in ["technicalSkills", "Technical Skills", "top_technical_skills", "top_10_technical_skills", "inDemandSkills"]:
        if key in market_data:
            tech_skills = market_data[key]
            break
    
    with col1:
        st.subheader("In-Demand Technical Skills")
        if tech_skills:
            if isinstance(tech_skills, list) and len(tech_skills) > 0:
                # Create DataFrame for visualization
                try:
                    skill_df = pd.DataFrame({
                        'Skill': tech_skills[:10],  # Limit to top 10 for visualization
                        'Demand': range(len(tech_skills[:10]), 0, -1)  # Higher demand for skills listed first
                    })
                    
                    fig = px.bar(skill_df, x='Demand', y='Skill', 
                                orientation='h', title="Technical Skills in Demand",
                                labels={'Demand': 'Relative Demand', 'Skill': ''},
                                color='Demand', color_continuous_scale='viridis')
                    
                    fig.update_layout(height=400, width=500)
                    st.plotly_chart(fig)
                except Exception as e:
                    # Fallback to simple list if visualization fails
                    st.write(f"Visualization error: {e}")  # Add error info for debugging
                    for i, skill in enumerate(tech_skills[:10]):
                        if isinstance(skill, str):
                            st.write(f"{i+1}. {skill}")
                        elif isinstance(skill, dict) and "name" in skill:
                            st.write(f"{i+1}. {skill['name']}")
            elif isinstance(tech_skills, dict):
                for key, value in tech_skills.items():
                    st.markdown(f"- **{key}**: {value}")
            else:
                st.write(tech_skills)
        else:
            st.info("No technical skills data available")
    
    # Soft Skills - handle different formats
    soft_skills = None
    
    # Try different possible keys - ADD YOUR SPECIFIC KEY
    for key in ["softSkills", "Soft Skills", "top_soft_skills", "top_5_soft_skills"]:
        if key in market_data:
            soft_skills = market_data[key]
            break
            
    with col2:
        # Rest of the code remains the same...
        st.subheader("Valued Soft Skills")
        if soft_skills:
            if isinstance(soft_skills, list) and len(soft_skills) > 0:
                try:
                    # Create pie chart for soft skills
                    fig = px.pie(names=soft_skills[:5], title="Soft Skills Distribution")
                    fig.update_layout(height=400, width=500)
                    fig.update_traces(textposition='inside', textinfo='label')
                    st.plotly_chart(fig)
                except Exception as e:
                    # Fallback to simple list if visualization fails
                    for skill in soft_skills[:5]:
                        if isinstance(skill, str):
                            st.write(f"- {skill}")
                        elif isinstance(skill, dict) and "name" in skill:
                            st.write(f"- {skill['name']}")
            elif isinstance(soft_skills, dict):
                for key, value in soft_skills.items():
                    st.markdown(f"- **{key}**: {value}")
            else:
                st.write(soft_skills)
        else:
            st.info("No soft skills data available")
    
    # Education Requirements - handle different formats
    education_req = None
    
    # Try different possible keys
    for key in ["educationRequirements", "Education Requirements"]:
        if key in market_data:
            education_req = market_data[key]
            break
            
    if education_req:
        st.subheader("Common Education Requirements")
        if isinstance(education_req, list):
            for req in education_req:
                st.write(f"- {req}")
        else:
            st.write(education_req)
    
    # Experience Level - handle different formats
    experience_level = None
    
    # Try different possible keys
    for key in ["experienceLevelExpectations", "Experience Level Expectations", "experienceLevel"]:
        if key in market_data:
            experience_level = market_data[key]
            break
            
    if experience_level:
        st.subheader("Experience Level Expectations")
        if isinstance(experience_level, list):
            for level in experience_level:
                st.write(f"- {level}")
        else:
            st.write(experience_level)
    
    # Industry Trends - handle different formats
    industry_trends = None
    
    # Try different possible keys
    for key in ["industryTrends", "Industry Trends"]:
        if key in market_data:
            industry_trends = market_data[key]
            break
            
    if industry_trends:
        st.subheader("Industry Trends")
        if isinstance(industry_trends, list):
            for trend in industry_trends:
                st.write(f"- {trend}")
        else:
            st.write(industry_trends)


# Function to display resume scoring
def display_resume_scoring(scoring_data, resume_data, market_data):
    st.header("📊 Resume Evaluation")
    
    # Add debug option
    debug_view = st.checkbox("Show raw data (debug)", key="debug_scoring", value=False)
    if debug_view:
        st.json(scoring_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Overall Score - handle different formats
        score = None
        
        # Try different possible keys
        for key in ["overallMatchScore", "Overall Match Score", "score", "match_score"]:
            if key in scoring_data:
                score = scoring_data[key]
                break
        
        if score is not None:
            # Convert score to number if it's a string
            if isinstance(score, str):
                try:
                    if score.isdigit():
                        score = int(score)
                    elif score.replace('.', '', 1).isdigit():
                        score = float(score)
                except:
                    pass
                
            st.subheader("Overall Market Match")
            
            # Display as gauge/progress bar if it's a number
            if isinstance(score, (int, float)) and 0 <= score <= 100:
                st.progress(score/100)
                st.markdown(f"<h1 style='text-align: center; color: {'green' if score >= 70 else 'orange' if score >= 50 else 'red'};'>{score}%</h1>", unsafe_allow_html=True)
            else:
                st.write(f"Score: {score}")
        else:
            st.subheader("Overall Match Score")
            st.info("Score not available")
    
    with col2:
        # Strengths and Improvement Areas - handle different formats
        strengths = None
        improvements = None
        
        # Try different possible keys for strengths
        for key in ["strengths", "Strengths"]:
            if key in scoring_data:
                strengths = scoring_data[key]
                break
                
        # Try different possible keys for improvement areas
        for key in ["improvementAreas", "Improvement Areas", "areasToImprove"]:
            if key in scoring_data:
                improvements = scoring_data[key]
                break
        
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            st.subheader("💪 Strengths")
            if strengths:
                if isinstance(strengths, list):
                    for strength in strengths:
                        if isinstance(strength, str):
                            st.markdown(f"- ✅ {strength}")
                        elif isinstance(strength, dict) and "text" in strength:
                            st.markdown(f"- ✅ {strength['text']}")
                elif isinstance(strengths, dict):
                    for key, value in strengths.items():
                        st.markdown(f"- ✅ **{key}**: {value}")
                else:
                    st.write(strengths)
            else:
                st.info("No strengths information available")
        
        with col2_2:
            st.subheader("🚀 Areas to Improve")
            if improvements:
                if isinstance(improvements, list):
                    for area in improvements:
                        if isinstance(area, str):
                            st.markdown(f"- ⚠️ {area}")
                        elif isinstance(area, dict) and "text" in area:
                            st.markdown(f"- ⚠️ {area['text']}")
                elif isinstance(improvements, dict):
                    for key, value in improvements.items():
                        st.markdown(f"- ⚠️ **{key}**: {value}")
                else:
                    st.write(improvements)
            else:
                st.info("No improvement areas information available")
    
    # Skills Match and Gaps - handle different formats
    st.subheader("Skills Analysis")
    col3, col4 = st.columns(2)
    
    skills_match = None
    skills_gaps = None
    
    # Try different possible keys for skills match
    for key in ["skillsMatch", "Skills Match", "matchedSkills"]:
        if key in scoring_data:
            skills_match = scoring_data[key]
            # Handle nested structures
            if isinstance(skills_match, dict) and "Matched Skills" in skills_match:
                skills_match = skills_match["Matched Skills"]
            break
            
    # Try different possible keys for skills gaps
    for key in ["skillsGaps", "Skills Gaps", "missingSkills"]:
        if key in scoring_data:
            skills_gaps = scoring_data[key]
            break
    
    with col3:
        st.markdown("**🎯 Skills Match**")
        if skills_match:
            if isinstance(skills_match, list):
                for skill in skills_match:
                    if isinstance(skill, str):
                        st.markdown(f"- ✓ {skill}")
                    elif isinstance(skill, dict) and "name" in skill:
                        st.markdown(f"- ✓ {skill['name']}")
            elif isinstance(skills_match, dict):
                for key, value in skills_match.items():
                    st.markdown(f"- ✓ **{key}**: {value}")
            else:
                st.write(skills_match)
        else:
            st.info("No matching skills found")
    
    with col4:
        st.markdown("**⚠️ Skills Gaps**")
        if skills_gaps:
            if isinstance(skills_gaps, list):
                for skill in skills_gaps:
                    if isinstance(skill, str):
                        st.markdown(f"- ✗ {skill}")
                    elif isinstance(skill, dict) and "name" in skill:
                        st.markdown(f"- ✗ {skill['name']}")
            elif isinstance(skills_gaps, dict):
                for key, value in skills_gaps.items():
                    st.markdown(f"- ✗ **{key}**: {value}")
            else:
                st.write(skills_gaps)
        else:
            st.info("No skill gaps identified")
    
    # Experience and Education Evaluation - handle different formats
    st.subheader("Experience & Education")
    col5, col6 = st.columns(2)
    
    experience_eval = None
    education_eval = None
    
    # Try different possible keys for experience evaluation
    for key in ["experienceEvaluation", "Experience Evaluation"]:
        if key in scoring_data:
            experience_eval = scoring_data[key]
            break
            
    # Try different possible keys for education evaluation
    for key in ["educationEvaluation", "Education Evaluation"]:
        if key in scoring_data:
            education_eval = scoring_data[key]
            break
    
    with col5:
        st.markdown("**Experience Evaluation**")
        if experience_eval:
            st.write(experience_eval)
        else:
            st.info("No experience evaluation available")
    
    with col6:
        st.markdown("**Education Evaluation**")
        if education_eval:
            st.write(education_eval)
        else:
            st.info("No education evaluation available")

# Function to display career guidance
def display_career_guidance(guidance_data):
    st.header("🧭 Personalized Career Guidance")
    
    # Show career goal if available
    if 'career_goal' in st.session_state and st.session_state.career_goal:
        st.info(f"**Your Career Goal:** {st.session_state.career_goal}")
    
    # Add debug option
    debug_view = st.checkbox("Show raw data (debug)", key="debug_guidance", value=False)
    if debug_view:
        st.json(guidance_data)
    
    # First check for Overall Assessment
    if "overallAssessment" in guidance_data:
        st.subheader("Overview")
        st.write(guidance_data["overallAssessment"])
        st.divider()
    
    # Career Path Recommendations - handle different formats
    career_paths = None
    
    # Try different possible keys
    for key in ["careerPathRecommendations", "Career Path Recommendations", "1. Career Path Recommendations"]:
        if key in guidance_data:
            career_paths = guidance_data[key]
            break
            
    if career_paths:
        st.subheader("Recommended Career Paths")
        
        if isinstance(career_paths, list):
            for i, path in enumerate(career_paths):
                if isinstance(path, dict):
                    # Try to extract title with different possible keys
                    title = None
                    for title_key in ["title", "path", "name", "Career Path", "Rank"]:
                        if title_key in path:
                            if title_key == "Rank":
                                # Special case for rank - get the Career Path value
                                title = path.get("Career Path", f"Career Path {path['Rank']}")
                            else:
                                title = path[title_key]
                            break
                    
                    if not title:
                        title = f"Career Path {i+1}"
                        
                    # Try to extract reasoning
                    reasoning = None
                    for reason_key in ["reasoning", "Reasoning", "description"]:
                        if reason_key in path:
                            reasoning = path[reason_key]
                            break
                    
                    with st.expander(f"**{title}**"):
                        if reasoning:
                            st.markdown("**Why this is a good fit:**")
                            st.write(reasoning)
                else:
                    st.write(f"- {path}")
        elif isinstance(career_paths, dict):
            for key, value in career_paths.items():
                st.subheader(key)
                st.write(value)
        else:
            st.write(career_paths)
    
    # Skill Development Plan - handle different formats
    skill_plan = None
    
    # Try different possible keys
    for key in ["skillDevelopmentPlan", "Skill Development Plan", "2. Skill Development Plan"]:
        if key in guidance_data:
            skill_plan = guidance_data[key]
            break
            
    if skill_plan:
        st.subheader("Skill Development Plan")
        
        if isinstance(skill_plan, list):
            for item in skill_plan:
                if isinstance(item, dict):
                    # Try to extract skill name with different possible keys
                    skill_name = None
                    for name_key in ["skill", "name", "Skill"]:
                        if name_key in item:
                            skill_name = item[name_key]
                            break
                    
                    if not skill_name:
                        skill_name = "Skill"
                    
                    # Extract description if available
                    description = item.get("Description", item.get("description", ""))
                        
                    # Try to extract resources
                    resources = item.get("Resources", item.get("resources", []))
                    
                    st.markdown(f"**{skill_name}**")
                    if description:
                        st.write(description)
                    
                    if resources:
                        st.markdown("**Resources:**")
                        if isinstance(resources, list):
                            for resource in resources:
                                st.markdown(f"- {resource}")
                        else:
                            st.write(resources)
                else:
                    st.write(f"- {item}")
        elif isinstance(skill_plan, dict):
            for key, value in skill_plan.items():
                st.subheader(key)
                if isinstance(value, list):
                    for item in value:
                        st.write(f"- {item}")
                else:
                    st.write(value)
        else:
            st.write(skill_plan)
    
    # Actions and Strategies
    col1, col2 = st.columns(2)
    
    # Short-term Actions - handle different formats
    short_term = None
    
    # Try different possible keys
    for key in ["shortTermActions", "Short Term Actions", "3. Short-Term Actions"]:
        if key in guidance_data:
            short_term = guidance_data[key]
            break
            
    with col1:
        if short_term:
            st.subheader("Immediate Actions (1-3 months)")
            if isinstance(short_term, list):
                for action in short_term:
                    st.markdown(f"- {action}")
            else:
                st.write(short_term)
    
    # Medium-term Strategy - handle different formats
    medium_term = None
    
    # Try different possible keys
    for key in ["mediumTermStrategy", "Medium Term Strategy", "4. Medium-Term Strategy"]:
        if key in guidance_data:
            medium_term = guidance_data[key]
            break
            
    with col2:
        if medium_term:
            st.subheader("Medium-term Strategy (1-2 years)")
            if isinstance(medium_term, list):
                for strategy in medium_term:
                    st.markdown(f"- {strategy}")
            else:
                st.write(medium_term)
    
    # Long-term Vision - handle different formats
    long_term = None
    
    # Try different possible keys
    for key in ["longTermVision", "Long Term Vision", "5. Long-Term Vision"]:
        if key in guidance_data:
            long_term = guidance_data[key]
            break
            
    if long_term:
        st.subheader("Long-term Vision (5+ years)")
        if isinstance(long_term, list):
            for vision in long_term:
                st.markdown(f"- {vision}")
        else:
            st.write(long_term)

# Function for the chatbot interface
def display_chatbot():
    st.header("💬 Career Advisor Chatbot")
    st.write("Ask me anything about your career, skills, or job market!")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask a question about your career...")
    
    if user_input and st.session_state.chatbot_agent:
        # Use a try-except block to handle potential errors
        try:
            # Display user message
            st.chat_message("user").write(user_input)
            
            # Add to history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Get response
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot_agent.get_response(
                    user_input, 
                    st.session_state.resume_data, 
                    st.session_state.career_guidance
                )
            
            # Display assistant response
            st.chat_message("assistant").write(response)
            
            # Add to history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
        except Exception as e:
            st.error(f"Error in chatbot response: {str(e)}")
            # Don't add the error to chat history

# Main function
def main():
    display_header()
    
    # Initialize API keys
    openai_key, serper_key, api_keys_provided = initialize_api_keys()
    
    # File upload
    uploaded_file = st.file_uploader("Upload your resume (PDF format)", type=["pdf"])
    
    if uploaded_file and api_keys_provided:
        # Get career goal from user
        career_goal = st.text_area(
            "What are your professional goals and aspirations?",
            help="For example: 'I want to transition to a senior data scientist role' or 'I'm looking to move into project management in the healthcare sector'",
            placeholder="Describe what you want to achieve in your professional life..."
        )
        
        # Process button with career goal check
        process_button = st.button("Analyze Resume")
        
        if process_button:
            if not career_goal or len(career_goal.strip()) < 10:
                st.warning("Please provide more details about your career goals for better personalized guidance.")
            else:
                # Save the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name
                
                # Process the resume with career goal
                file_name = uploaded_file.name
                # Store career goal in session state
                st.session_state.career_goal = career_goal
                
                # Process the resume and store the data in session state
                st.session_state.resume_data, st.session_state.market_data, st.session_state.scoring_data, st.session_state.career_guidance = process_resume(temp_file_path, openai_key, serper_key, career_goal)
                
                # Store the name of the processed file
                st.session_state.last_processed_file = file_name
                
                # Clean up the temporary file
                os.unlink(temp_file_path)
        
        # Check if resume was successfully processed
        if 'resume_data' in st.session_state and st.session_state.resume_data:
            # Display tabs for different sections
            tabs = st.tabs(["📄 Resume Analysis", "🌐 Market Trends", "📊 Resume Score", "🧭 Career Guidance", "💬 Chat"])
            
            with tabs[0]:
                display_resume_analysis(st.session_state.resume_data)
                
            with tabs[1]:
                if st.session_state.market_data:
                    display_market_trends(st.session_state.market_data)
                else:
                    st.warning("Market data not available.")
                    
            with tabs[2]:
                if st.session_state.scoring_data and st.session_state.resume_data and st.session_state.market_data:
                    display_resume_scoring(st.session_state.scoring_data, st.session_state.resume_data, st.session_state.market_data)
                else:
                    st.warning("Resume scoring data not available.")
                    
            with tabs[3]:
                if st.session_state.career_guidance:
                    display_career_guidance(st.session_state.career_guidance)
                else:
                    st.warning("Career guidance data not available.")
            
            with tabs[4]:
                if st.session_state.chatbot_agent:
                    display_chatbot()
                else:
                    st.warning("Chatbot not initialized.")
        else:
            st.error("Unable to process the resume. Please try again.")
            
    elif uploaded_file and not api_keys_provided:
        st.warning("Please provide API keys in the sidebar to analyze your resume")
        
    else:
        # Display sample images to guide users
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Resume Analysis")
            st.image("https://img.icons8.com/color/240/000000/parse-from-clipboard.png", width=100)
            st.write("Extract key skills and experience from your resume")
            
        with col2:
            st.subheader("Market Alignment")
            st.image("https://img.icons8.com/color/240/000000/job.png", width=100)
            st.write("See how your skills match current market demands")
            
        with col3:
            st.subheader("Career Guidance")
            st.image("https://img.icons8.com/color/240/000000/compass.png", width=100)
            st.write("Get personalized recommendations for your career path")
    
    # Info about API keys at the bottom
    st.sidebar.divider()
    st.sidebar.subheader("About API Keys")
    st.sidebar.info(
        """
        This app requires:
        1. **OpenAI API Key** - For AI-based analysis ([Get one here](https://platform.openai.com/))
        2. **Serper API Key** - For job market data ([Get one here](https://serper.dev/))
        
        Your API keys are used only for this session and are not stored.
        """
    )
    
    # Footer
    st.sidebar.divider()
    st.sidebar.markdown("Made with ❤️ using CrewAI & Streamlit")

# Run the app
if __name__ == "__main__":
    main()
