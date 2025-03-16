import requests
import json
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import os

class SerperScraper:
    """Utility class to scrape job market data using Serper API."""
    
    def __init__(self, api_key: str):
        """
        Initialize with Serper API key.
        
        Args:
            api_key: Serper API key
        """
        load_dotenv()
        # Use the provided API key rather than trying to get from .env
        self.api_key = api_key
        self.base_url = "https://google.serper.dev/search"
        self.headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
    
    def search_jobs(self, query: str, location: Optional[str] = None, 
                    num_results: int = 20) -> Dict[str, Any]:
        """
        Search for jobs using the Serper API.
        
        Args:
            query: Job search query (e.g., "data scientist")
            location: Location for job search (e.g., "New York")
            num_results: Number of results to return
            
        Returns:
            Dictionary containing search results
        """
        search_query = query
        if location:
            search_query += f" jobs in {location}"
        else:
            search_query += " jobs"
            
        payload = {
            "q": search_query,
            "num": num_results
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error searching jobs with Serper API: {str(e)}")
            return {"error": str(e)}
    
    def extract_job_details(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract job details from search results.
        
        Args:
            search_results: Search results from Serper API
            
        Returns:
            List of job details
        """
        job_listings = []
        
        try:
            # Extract organic search results
            if "organic" in search_results:
                for result in search_results["organic"]:
                    job = {
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                        "source": result.get("source", "")
                    }
                    job_listings.append(job)
            
            return job_listings
        except Exception as e:
            print(f"Error extracting job details: {str(e)}")
            return []
    
    def search_multiple_job_titles(self, job_titles: List[str], 
                                location: Optional[str] = None,
                                delay: int = 2) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for multiple job titles with delay between requests to avoid rate limiting.
        
        Args:
            job_titles: List of job titles to search for
            location: Location for job search
            delay: Delay between requests in seconds
            
        Returns:
            Dictionary mapping job titles to their search results
        """
        results = {}
        
        for job_title in job_titles:
            search_results = self.search_jobs(job_title, location)
            if "error" not in search_results:
                job_listings = self.extract_job_details(search_results)
                results[job_title] = job_listings
            else:
                results[job_title] = []
            
            # Add delay to avoid rate limiting
            time.sleep(delay)
            
        return results
