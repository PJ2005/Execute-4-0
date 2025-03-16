import logging
import json
import re
from typing import Dict, Any, List
import os
from dotenv import load_dotenv
import requests
from utils.serper_scraper import SerperScraper

# Configure logging
logging.basicConfig(level=logging.INFO)

class JobMarketService:
    """Service for accessing job market data and insights."""
    
    def __init__(self, serper_api_key=None, openrouter_api_key=None):
        """
        Initialize JobMarketService with necessary API keys.
        
        Args:
            serper_api_key: API key for Serper
            openrouter_api_key: API key for OpenRouter LLM
        """
        load_dotenv()
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        
        if not self.serper_api_key:
            logging.warning("Serper API key not found")
            
        if not self.openrouter_api_key:
            logging.warning("OpenRouter API key not found")
            
        # Initialize scrapers
        self.serper_scraper = SerperScraper(self.serper_api_key) if self.serper_api_key else None
        
    def get_job_market_comparison(self, job_title: str, location: str = "",
                               required_skills: List[str] = None) -> Dict[str, Any]:
        """
        Get job market comparison data for specific job title and skills.
        
        Args:
            job_title: Job title to analyze
            location: Optional location for job search
            required_skills: List of skills to analyze their demand
            
        Returns:
            Dictionary with market insights and skill demand
        """
        if not self.serper_scraper:
            return {"error": "Serper API key not configured"}
            
        try:
            # Get market insights
            market_insights = self.serper_scraper.get_job_market_insights(job_title, location)
            
            # Get skill demand analysis if skills provided
            skills_demand = {}
            if required_skills:
                skills_demand = self.serper_scraper.analyze_skill_demand(
                    required_skills, job_title
                )
                
            # If there's no skill demand data, try to extract from LLM analysis
            if not skills_demand and self.openrouter_api_key and required_skills:
                skills_summary = ", ".join(required_skills)
                prompt = f"""
                Analyze the current market demand for these skills for {job_title} positions:
                {skills_summary}
                
                For each skill, classify its demand as "high", "medium", or "low".
                
                Return your analysis ONLY as a JSON object where each key is a skill 
                and the value is its demand level. For example:
                {{
                  "Python": "high",
                  "JavaScript": "medium",
                  "Fortran": "low"
                }}
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
                                {"role": "system", "content": "You are an expert in job market analysis."},
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": 0.1
                        }
                    )
                    
                    analysis_text = response.json()["choices"][0]["message"]["content"]
                    
                    # Extract JSON if present
                    import re
                    import json
                    json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                    
                    if json_match:
                        try:
                            skills_demand = json.loads(json_match.group(0))
                        except json.JSONDecodeError:
                            skills_demand = {skill: "unknown" for skill in required_skills}
                    else:
                        skills_demand = {skill: "unknown" for skill in required_skills}
                        
                except Exception as e:
                    skills_demand = {skill: "unknown" for skill in required_skills}
                
            return {
                "job_title": job_title,
                "location": location if location else "Global",
                "market_summary": market_insights.get("summary", "No data available"),
                "skills_demand": skills_demand,
                "salary_info": market_insights.get("salary_info", "No salary data available"),
                "demand_level": market_insights.get("demand_level", "unknown")
            }
            
        except Exception as e:
            logging.error(f"Error getting job market comparison: {e}")
            return {
                "error": str(e),
                "job_title": job_title,
                "market_summary": f"Failed to retrieve market data: {str(e)}",
                "skills_demand": {skill: "unknown" for skill in required_skills} if required_skills else {}
            }
            
    def get_trending_skills(self, job_title: str, location: str = "") -> Dict[str, Any]:
        """
        Get trending skills for a specific job title.
        
        Args:
            job_title: Job title to analyze
            location: Optional location parameter
            
        Returns:
            Dictionary with trending skills information
        """
        if not self.serper_scraper:
            return {"error": "Serper API key not configured"}
            
        if not self.openrouter_api_key:
            return {"error": "OpenRouter API key not configured"}
            
        try:
            # Get job search results
            job_results = self.serper_scraper.search_jobs(job_title, location)
            job_listings = self.serper_scraper.extract_job_details(job_results)
            
            # Extract text from listings
            listings_text = ""
            for job in job_listings[:5]:  # Use top 5 listings
                listings_text += f"Job Title: {job.get('title', '')}\n"
                if 'description' in job:
                    listings_text += f"Description: {job.get('description', '')}\n\n"
                else:
                    listings_text += f"Snippet: {job.get('snippet', '')}\n\n"
            
            # Use LLM to analyze trending skills
            prompt = f"""
            Based on these job listings for {job_title} positions, identify:
            
            1. Top 10 most in-demand technical skills
            2. Top 5 most in-demand soft skills
            3. Education requirements that stand out
            4. Experience level expectations
            5. Any emerging technology trends
            
            Job Listings:
            {listings_text}
            
            Format your response as a valid JSON object with the following structure:
            {{
              "technical_skills": ["Skill 1", "Skill 2", ...],
              "soft_skills": ["Skill 1", "Skill 2", ...],
              "education": "Description of education requirements",
              "experience": "Description of experience requirements",
              "trends": ["Trend 1", "Trend 2", ...]
            }}
            
            Return ONLY valid JSON.
            """
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "google/gemma-3-4b-it:free",
                    "messages": [
                        {"role": "system", "content": "You are an expert job market analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1
                }
            )
            
            analysis_text = response.json()["choices"][0]["message"]["content"]
            
            # Extract JSON
            from utils.json_handler import JsonHandler
            trend_data = JsonHandler.extract_json(analysis_text)
            
            if not trend_data:
                trend_data = {
                    "technical_skills": ["Data not available"],
                    "soft_skills": ["Data not available"],
                    "education": "Data not available",
                    "experience": "Data not available",
                    "trends": ["Data not available"]
                }
                
            # Add market insights
            market_insights = self.serper_scraper.get_job_market_insights(job_title, location)
            
            return {
                "job_title": job_title,
                "location": location if location else "Global",
                "market_summary": market_insights.get("summary", "No market summary available"),
                "trending_skills": trend_data.get("technical_skills", []),
                "soft_skills": trend_data.get("soft_skills", []),
                "education_requirements": trend_data.get("education", "Not specified"),
                "experience_requirements": trend_data.get("experience", "Not specified"),
                "emerging_trends": trend_data.get("trends", []),
                "salary_info": market_insights.get("salary_info", "Not available"),
                "demand_level": market_insights.get("demand_level", "unknown")
            }
            
        except Exception as e:
            logging.error(f"Error getting trending skills: {e}")
            return {
                "error": str(e),
                "job_title": job_title,
                "trending_skills": ["Error retrieving data"],
                "market_summary": f"Failed to retrieve market data: {str(e)}"
            }
