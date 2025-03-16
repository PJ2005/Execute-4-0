import requests
import os
import json
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)

class SerperScraper:
    """
    Helper class for scraping job market data using Serper API.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the SerperScraper with API key.
        
        Args:
            api_key: The Serper API key
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError("Serper API key not found")
        
        self.base_url = "https://serpapi.com/search"
        
    def search_jobs(self, job_title: str, location: str = "", limit: int = 10) -> Dict[str, Any]:
        """
        Search for job listings.
        
        Args:
            job_title: The job title to search for
            location: Optional location parameter
            limit: Maximum number of results
            
        Returns:
            Search results from Serper API
        """
        try:
            search_query = f"{job_title} jobs"
            if location:
                search_query += f" in {location}"
                
            response = requests.post(
                "https://google.serper.dev/search",
                headers={'X-API-KEY': self.api_key},
                json={
                    'q': search_query,
                    'gl': 'us',
                    'num': limit
                }
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logging.error(f"Error searching jobs: {e}")
            return {"error": str(e)}
    
    def extract_job_details(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract job details from search results.
        
        Args:
            search_results: The Serper search results
            
        Returns:
            List of job listing details
        """
        job_listings = []
        
        try:
            # Extract organic search results
            if "organic" in search_results:
                for result in search_results["organic"]:
                    job_listing = {
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                        "source": "organic"
                    }
                    job_listings.append(job_listing)
            
            # Extract job results if available
            if "jobResults" in search_results:
                for job in search_results["jobResults"]:
                    job_details = {
                        "title": job.get("title", ""),
                        "company": job.get("company", ""),
                        "location": job.get("location", ""),
                        "description": job.get("description", ""),
                        "salary": job.get("salary", ""),
                        "link": job.get("link", ""),
                        "source": "job_listing"
                    }
                    job_listings.append(job_details)
            
            return job_listings
            
        except Exception as e:
            logging.error(f"Error extracting job details: {e}")
            return []
    
    def get_job_market_insights(self, job_title: str, location: str = "") -> Dict[str, Any]:
        """
        Get job market insights including salary and demand information.
        
        Args:
            job_title: The job title to analyze
            location: Optional location parameter
            
        Returns:
            Dictionary with job market insights
        """
        try:
            search_query = f"{job_title} average salary trends skills demand"
            if location:
                search_query += f" in {location}"
                
            response = requests.post(
                "https://google.serper.dev/search",
                headers={'X-API-KEY': self.api_key},
                json={
                    'q': search_query,
                    'gl': 'us',
                    'num': 5
                }
            )
            
            response.raise_for_status()
            results = response.json()
            
            # Extract insights from snippets and knowledge graph
            insights = {
                "summary": f"Market insights for {job_title} positions",
                "salary_info": "No salary information available",
                "demand_level": "Unknown"
            }
            
            # Check knowledge graph for salary
            if "knowledgeGraph" in results:
                kg = results["knowledgeGraph"]
                if "salary" in kg:
                    insights["salary_info"] = kg["salary"]
                if "description" in kg:
                    insights["summary"] = kg["description"]
            
            # Extract from snippets
            if "organic" in results:
                snippets = [result.get("snippet", "") for result in results["organic"]]
                combined_text = " ".join(snippets)
                
                # Look for salary patterns
                salary_pattern = r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?(?:\s*-\s*\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?)?(?:\s*per\s*year)?'
                import re
                salary_matches = re.findall(salary_pattern, combined_text)
                if salary_matches:
                    insights["salary_info"] = salary_matches[0]
                
                # Check for demand indicators
                demand_keywords = {
                    "high": ["high demand", "growing rapidly", "shortage", "competitive", "seeking", "urgently"],
                    "medium": ["steady demand", "stable", "moderate growth"],
                    "low": ["declining", "saturated", "low demand", "competitive market", "difficult"]
                }
                
                for level, keywords in demand_keywords.items():
                    if any(keyword in combined_text.lower() for keyword in keywords):
                        insights["demand_level"] = level
                        break
            
            return insights
            
        except Exception as e:
            logging.error(f"Error getting job market insights: {e}")
            return {"error": str(e)}
            
    def analyze_skill_demand(self, required_skills: List[str], job_title: str) -> Dict[str, str]:
        """
        Analyze the demand for specific skills related to a job title.
        
        Args:
            required_skills: List of skills to analyze
            job_title: The job title for context
            
        Returns:
            Dictionary mapping skills to their demand level (high, medium, low)
        """
        skills_demand = {}
        
        try:
            for skill in required_skills:
                search_query = f"{skill} demand in {job_title} jobs"
                
                response = requests.post(
                    "https://google.serper.dev/search",
                    headers={'X-API-KEY': self.api_key},
                    json={
                        'q': search_query,
                        'gl': 'us',
                        'num': 3
                    }
                )
                
                response.raise_for_status()
                results = response.json()
                
                # Extract from snippets
                demand_level = "unknown"
                if "organic" in results:
                    snippets = [result.get("snippet", "").lower() for result in results["organic"]]
                    combined_text = " ".join(snippets)
                    
                    # Check for demand indicators
                    if any(keyword in combined_text for keyword in ["high demand", "highly sought", "in-demand", "top skill"]):
                        demand_level = "high"
                    elif any(keyword in combined_text for keyword in ["growing demand", "increasingly", "valuable"]):
                        demand_level = "medium"
                    elif any(keyword in combined_text for keyword in ["basic skill", "expected", "common", "standard"]):
                        demand_level = "medium"
                    elif any(keyword in combined_text for keyword in ["declining", "outdated", "less important"]):
                        demand_level = "low"
                    else:
                        demand_level = "medium"  # Default to medium if no clear indicators
                
                skills_demand[skill] = demand_level
                
                # Sleep briefly to avoid API rate limits
                import time
                time.sleep(0.5)
                
            return skills_demand
            
        except Exception as e:
            logging.error(f"Error analyzing skill demand: {e}")
            return {skill: "unknown" for skill in required_skills}
