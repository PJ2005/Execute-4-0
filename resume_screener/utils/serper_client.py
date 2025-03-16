import os
import requests
import json
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

class SerperClient:
    """Client for interacting with the Serper API for job market data."""
    
    def __init__(self):
        self.api_key = os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError("Serper API key not found. Please set SERPER_API_KEY in .env file")
        
        self.base_url = "https://google.serper.dev/search"
        self.headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
    
    def search_job_market_data(self, job_title: str, location: str = "", skills: List[str] = None) -> Dict[str, Any]:
        """
        Search for job market data related to a specific job title.
        
        Args:
            job_title: The job title to search for
            location: Optional location to focus the search
            skills: Optional list of skills to include in the search
            
        Returns:
            Dictionary containing search results
        """
        query = f"{job_title} job market demand"
        if location:
            query += f" in {location}"
        if skills and len(skills) > 0:
            query += f" skills: {', '.join(skills[:3])}"  # Limit to 3 skills for better results
            
        try:
            payload = json.dumps({
                "q": query,
                "num": 10  # Number of results to return
            })
            
            response = requests.post(self.base_url, headers=self.headers, data=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error calling Serper API: {e}")
            return {"error": str(e)}
    
    def extract_job_insights(self, job_title: str, location: str = "", skills: List[str] = None) -> Dict[str, Any]:
        """
        Extract meaningful job market insights for a given job title.
        
        Args:
            job_title: The job title to analyze
            location: Optional location to focus the analysis
            skills: Optional list of skills to include
            
        Returns:
            Dictionary containing job market insights
        """
        search_results = self.search_job_market_data(job_title, location, skills)
        
        if "error" in search_results:
            return {"error": search_results["error"]}
        
        insights = {
            "job_title": job_title,
            "location": location if location else "Global",
            "top_results": [],
            "source_urls": []
        }
        
        # Extract organic search results
        if "organic" in search_results:
            for result in search_results["organic"][:5]:  # Top 5 results
                insights["top_results"].append({
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", "")
                })
                insights["source_urls"].append(result.get("link", ""))
        
        return insights
