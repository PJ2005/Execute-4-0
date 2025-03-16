from crewai import Agent, Task
import requests
from typing import Dict, Any, List
import json
from dotenv import load_dotenv
import os

class JobMarketAnalyzerAgent:
    """Agent responsible for analyzing the job market and identifying trends"""
    
    def __init__(self, openrouter_api_key: str, serper_scraper=None):
        """
        Initialize the JobMarketAnalyzerAgent.
        
        Args:
            openrouter_api_key: OpenRouter API key for model access
            serper_scraper: SerperScraper instance for job market scraping
        """
        self.openrouter_api_key = openrouter_api_key
        
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key not provided")
        
        self.serper_scraper = serper_scraper
        
        # Create the CrewAI agent
        self.agent = Agent(
            role="Job Market Research Specialist",
            goal="Identify current job market trends and in-demand skills",
            backstory=(
                "With a background in labor economics and recruitment analytics, you "
                "specialize in analyzing job markets to identify emerging trends. "
                "You have a deep understanding of how skills are valued across different "
                "industries and positions."
            ),
            verbose=True,
            allow_delegation=False,
            llm_config={"model": "google/gemma-3-12b-it:free"}
        )
    
    def analyze_job_listings(self, job_data: List[Dict[str, Any]], 
                          job_title: str) -> Dict[str, Any]:
        """
        Analyze job listings to extract trending skills and requirements.
        
        Args:
            job_data: List of job listings data from Serper
            job_title: Job title being researched
            
        Returns:
            Dictionary with analysis of in-demand skills and trends
        """
        # Convert job data to a string for the prompt
        job_listings_str = json.dumps(job_data[:10])  # Limit to first 10 to avoid token limits
        
        prompt = f"""
        Analyze these job listings for "{job_title}" positions and extract:
        
        1. Top 10 technical skills in demand (programming languages, tools, platforms)
        2. Top 5 soft skills mentioned
        3. Common education requirements
        4. Experience level expectations (junior, mid-level, senior)
        5. Salary ranges (if available)
        6. Industry trends visible from these listings
        
        Job Listings Data:
        {job_listings_str}
        
        You MUST return ONLY a valid JSON object with the following format:
        {{
            "top_10_technical_skills": ["Skill 1", "Skill 2", "Skill 3", ...],
            "top_5_soft_skills": ["Skill 1", "Skill 2", "Skill 3", ...],
            "common_education_requirements": "Description of education requirements",
            "experience_level_expectations": "Description of experience expectations",
            "salary_ranges": "Description of salary ranges",
            "industry_trends": ["Trend 1", "Trend 2", "Trend 3", ...]
        }}
        
        Make sure to provide ONLY the JSON object with no additional text or explanations.
        """
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "google/gemma-3-4b-it:free",
                    "messages": [
                        {"role": "system", "content": "You are a job market analysis expert who extracts key insights from job listings. Output ONLY valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1
                }
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            # Extract the response content
            if "choices" in response_data and len(response_data["choices"]) > 0:
                result = response_data["choices"][0]["message"]["content"]
                
                # Log the raw result for debugging
                print("Raw API response from job analysis:", result)
                
                # Process the response to extract JSON
                json_content = None
                
                # Look for content in JSON code blocks
                if "```json" in result:
                    json_parts = result.split("```json")
                    if len(json_parts) > 1:
                        json_block = json_parts[1].split("```")[0].strip()
                        json_content = json_block
                
                # If not found in JSON code block, look for general code blocks
                elif "```" in result:
                    json_parts = result.split("```")
                    if len(json_parts) > 1:
                        json_block = json_parts[1].strip()
                        json_content = json_block
                
                # If not found in code blocks, look for JSON structure directly
                else:
                    # Try to find content between curly braces
                    start_idx = result.find('{')
                    end_idx = result.rfind('}')
                    if (start_idx != -1 and end_idx != -1):
                        json_content = result[start_idx:end_idx+1].strip()
                    else:
                        # If no JSON structure found, use the full content
                        json_content = result.strip()
                
                try:
                    # Try to parse the extracted content as JSON
                    if json_content:
                        parsed_analysis = json.loads(json_content)
                        
                        # Ensure expected keys exist
                        expected_keys = [
                            "top_10_technical_skills", 
                            "top_5_soft_skills", 
                            "common_education_requirements", 
                            "experience_level_expectations", 
                            "salary_ranges", 
                            "industry_trends"
                        ]
                        
                        for key in expected_keys:
                            if key not in parsed_analysis:
                                if key.endswith("_skills"):
                                    parsed_analysis[key] = ["No specific skills identified"]
                                elif key == "industry_trends":
                                    parsed_analysis[key] = ["No specific trends identified"]
                                else:
                                    parsed_analysis[key] = "Information not available in job listings"
                        
                        return parsed_analysis
                    
                    else:
                        raise ValueError("Could not extract JSON content from the response")
                    
                except json.JSONDecodeError as json_err:
                    print(f"JSONDecodeError: {json_err}")
                    print("Content that failed to parse:", json_content)
                    raise
                    
            else:
                raise ValueError("Unexpected API response format: missing 'choices' field")
                    
        except Exception as e:
            print(f"Error in job market analysis: {str(e)}")
            
            # Return a fallback structure
            return {
                "top_10_technical_skills": ["Unable to analyze technical skills due to an error"],
                "top_5_soft_skills": ["Unable to analyze soft skills due to an error"],
                "common_education_requirements": "Unable to analyze education requirements due to an error",
                "experience_level_expectations": "Unable to analyze experience expectations due to an error",
                "salary_ranges": "Unable to analyze salary ranges due to an error",
                "industry_trends": ["Unable to analyze industry trends due to an error"],
                "error": str(e)
            }
    
    def create_job_market_task(self, job_title: str, location: str = None) -> Task:
        """
        Create a CrewAI Task for job market analysis.
        
        Args:
            job_title: Job title to research
            location: Optional location for targeted analysis
            
        Returns:
            CrewAI Task object
        """
        location_str = f" in {location}" if location else ""
        return Task(
            description=f"Research the job market for {job_title} positions{location_str} and identify trending skills and requirements",
            expected_output="Structured analysis of in-demand skills and job market trends",
            agent=self.agent
        )
    
    def get_trending_skills_for_profile(self, resume_data: Dict[str, Any], career_goal: str = None) -> Dict[str, Any]:
        """
        Get trending skills based on a candidate's resume profile and career goal.
        
        Args:
            resume_data: Resume data extracted by ResumeAnalyzerAgent
            career_goal: Optional career goal specified by the user
            
        Returns:
            Dictionary with relevant job market analysis
        """
        # Extract potential job titles from resume or career goal
        current_title = ""
        
        # If career goal is provided, try to extract a job title from it
        if career_goal:
            prompt = f"""
            Based on this career goal: "{career_goal}"
            What would be the most relevant job title to search for? Return only a single job title, no explanation.
            """
            
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openrouter_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "google/gemma-3-4b-it:free",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 20
                    }
                )
                
                response.raise_for_status()
                response_data = response.json()
                current_title = response_data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"Error extracting job title from career goal: {str(e)}")
                # Fall back to resume extraction if this fails
        
        # If no title found from career goal, extract from resume
        if not current_title and "workExperience" in resume_data and len(resume_data["workExperience"]) > 0:
            current_title = resume_data["workExperience"][0].get("jobTitle", "")
        
        if not current_title and "personalInformation" in resume_data:
            current_title = resume_data["personalInformation"].get("title", "")
        
        # If still no title, generate from skills
        if not current_title and "skills" in resume_data and resume_data["skills"]:
            skills_list = ", ".join(resume_data["skills"][:5])
            
            prompt = f"""
            Based on these skills: {skills_list}
            What would be the most likely job title for this person?
            Return only a single job title, no explanation.
            """
            
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openrouter_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "google/gemma-3-4b-it:free",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 20
                    }
                )
                
                response.raise_for_status()
                response_data = response.json()
                current_title = response_data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"Error generating job title from skills: {str(e)}")
                current_title = "Professional"
        
        # Use the Serper scraper to get job listings
        if self.serper_scraper:
            # If career goal exists, include it in the analysis
            career_context = ""
            if career_goal:
                career_context = f"Also consider that the user's career goal is: '{career_goal}'"
                
            job_results = self.serper_scraper.search_jobs(current_title)
            job_listings = self.serper_scraper.extract_job_details(job_results)
            
            # Include career goal in analysis if available
            if career_goal:
                return self.analyze_job_listings_with_goal(job_listings, current_title, career_goal)
            else:
                return self.analyze_job_listings(job_listings, current_title)
        else:
            return {"error": "Serper scraper not initialized"}
            
    def analyze_job_listings_with_goal(self, job_data: List[Dict[str, Any]], 
                                     job_title: str, career_goal: str) -> Dict[str, Any]:
        """
        Analyze job listings with consideration for the user's career goal.
        
        Args:
            job_data: List of job listings data from Serper
            job_title: Job title being researched
            career_goal: User's stated career goal
            
        Returns:
            Dictionary with analysis of in-demand skills and trends
        """
        # Convert job data to a string for the prompt
        job_listings_str = json.dumps(job_data[:10])  # Limit to first 10 to avoid token limits
        
        prompt = f"""
        Analyze these job listings for "{job_title}" positions and extract:
        
        1. Top 10 technical skills in demand (programming languages, tools, platforms)
        2. Top 5 soft skills mentioned
        3. Common education requirements
        4. Experience level expectations (junior, mid-level, senior)
        5. Salary ranges (if available)
        6. Industry trends visible from these listings
        
        User's Career Goal: "{career_goal}"
        
        Focus your analysis on how these job listings align with the user's stated career goal. 
        Emphasize skills and qualifications that would be most relevant for the user to achieve their career aspirations.
        
        Job Listings Data:
        {job_listings_str}
        
        You MUST return ONLY a valid JSON object with the following format:
        {{
            "top_10_technical_skills": ["Skill 1", "Skill 2", "Skill 3", ...],
            "top_5_soft_skills": ["Skill 1", "Skill 2", "Skill 3", ...],
            "common_education_requirements": "Description of education requirements",
            "experience_level_expectations": "Description of experience expectations",
            "salary_ranges": "Description of salary ranges",
            "industry_trends": ["Trend 1", "Trend 2", "Trend 3", ...],
            "career_path_relevance": "Explanation of how these trends align with the user's career goal"
        }}
        
        Make sure to provide ONLY the JSON object with no additional text or explanations.
        """
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "google/gemma-3-4b-it:free",
                    "messages": [
                        {"role": "system", "content": "You are a job market analysis expert who extracts key insights from job listings with consideration for career goals. Output ONLY valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1
                }
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            # Extract the response content
            if "choices" in response_data and len(response_data["choices"]) > 0:
                result = response_data["choices"][0]["message"]["content"]
                
                # Log the raw result for debugging
                print("Raw API response from job analysis with career goal:", result)
                
                # Process the response to extract JSON
                json_content = None
                
                # Look for content in JSON code blocks
                if "```json" in result:
                    json_parts = result.split("```json")
                    if len(json_parts) > 1:
                        json_block = json_parts[1].split("```")[0].strip()
                        json_content = json_block
                
                # If not found in JSON code block, look for general code blocks
                elif "```" in result:
                    json_parts = result.split("```")
                    if len(json_parts) > 1:
                        json_block = json_parts[1].strip()
                        json_content = json_block
                
                # If not found in code blocks, look for JSON structure directly
                else:
                    # Try to find content between curly braces
                    start_idx = result.find('{')
                    end_idx = result.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_content = result[start_idx:end_idx+1].strip()
                    else:
                        # If no JSON structure found, use the full content
                        json_content = result.strip()
                
                try:
                    # Try to parse the extracted content as JSON
                    if json_content:
                        parsed_analysis = json.loads(json_content)
                        
                        # Ensure expected keys exist
                        expected_keys = [
                            "top_10_technical_skills", 
                            "top_5_soft_skills", 
                            "common_education_requirements", 
                            "experience_level_expectations", 
                            "salary_ranges", 
                            "industry_trends",
                            "career_path_relevance"
                        ]
                        
                        for key in expected_keys:
                            if key not in parsed_analysis:
                                if key.endswith("_skills"):
                                    parsed_analysis[key] = ["No specific skills identified"]
                                elif key == "industry_trends":
                                    parsed_analysis[key] = ["No specific trends identified"]
                                elif key == "career_path_relevance":
                                    parsed_analysis[key] = "No specific relevance identified"
                                else:
                                    parsed_analysis[key] = "Information not available in job listings"
                        
                        return parsed_analysis
                    
                    else:
                        raise ValueError("Could not extract JSON content from the response")
                    
                except json.JSONDecodeError as json_err:
                    print(f"JSONDecodeError: {json_err}")
                    print("Content that failed to parse:", json_content)
                    raise
                    
            else:
                raise ValueError("Unexpected API response format: missing 'choices' field")
                    
        except Exception as e:
            print(f"Error in job market analysis: {str(e)}")
            
            # Return a fallback structure
            return {
                "top_10_technical_skills": ["Unable to analyze technical skills due to an error"],
                "top_5_soft_skills": ["Unable to analyze soft skills due to an error"],
                "common_education_requirements": "Unable to analyze education requirements due to an error",
                "experience_level_expectations": "Unable to analyze experience expectations due to an error",
                "salary_ranges": "Unable to analyze salary ranges due to an error",
                "industry_trends": ["Unable to analyze industry trends due to an error"],
                "career_path_relevance": "Unable to analyze career path relevance due to an error",
                "error": str(e)
            }
