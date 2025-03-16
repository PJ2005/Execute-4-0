import json
import re
import os
import requests
from crewai import Agent
from typing import Dict, Any, List
from dotenv import load_dotenv

from utils.pdf_parser import PDFParser

load_dotenv()

class ResumeProcessorAgent:
    """Agent responsible for processing and extracting structured data from resumes."""
    
    def __init__(self, openrouter_api_key=None):
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key not found. Please set OPENROUTER_API_KEY in .env file")
            
        self.pdf_parser = PDFParser()
        self.agent = self._create_agent()
        
    def _create_agent(self) -> Agent:
        """Create the CrewAI agent for resume processing."""
        return Agent(
            role="Resume Analyzer Expert",
            goal="Extract structured data from resumes with high accuracy",
            backstory=(
                "You are an expert in parsing and extracting information from resumes. "
                "You can identify skills, experience, education, and personal details "
                "from unstructured resume text to enable effective matching with job requirements."
            ),
            verbose=True,
            allow_delegation=False,
            llm_config={"model": "google/gemma-3-12b-it:free"}
        )
    
    def process_resume(self, resume_text: str) -> Dict[str, Any]:
        """
        Extract structured information from a resume.
        
        Args:
            resume_text: The resume text content
            
        Returns:
            Dictionary with structured resume information
        """
        prompt = f"""
        Extract key information from the following resume text. Include:
        
        1. Personal Information (name, email, phone, location)
        2. Skills (technical and soft skills)
        3. Experience (job titles, companies, dates, descriptions)
        4. Education (degrees, institutions, dates)
        5. Certifications (if any)
        6. Languages (if any)
        7. Summary/Objective (if any)
        
        Format your response as a valid JSON with the following structure:
        {{
            "name": "Full Name",
            "contact_info": {{
                "email": "email address",
                "phone": "phone number",
                "location": "city, state"
            }},
            "skills": ["Skill 1", "Skill 2", ...],
            "experience": [
                {{
                    "title": "Job Title",
                    "company": "Company Name",
                    "start_date": "Start Date",
                    "end_date": "End Date or Present",
                    "duration": "Duration",
                    "description": "Job Description"
                }}
            ],
            "education": [
                {{
                    "degree": "Degree Name",
                    "institution": "Institution Name",
                    "field": "Field of Study",
                    "graduation_date": "Graduation Date"
                }}
            ],
            "certifications": ["Certification 1", "Certification 2", ...],
            "languages": ["Language 1", "Language 2", ...],
            "summary": "Professional summary or objective statement"
        }}
        
        Resume Text:
        {resume_text[:3000]}  # Limit text to avoid token limits
        
        Return ONLY valid JSON.
        """
        
        try:
            response = self._analyze_text(f"{prompt}\n\nRESUME:\n{resume_text}", temperature=0.1)
            
            # Try to extract JSON from the response
            json_response = self._extract_json(response)
            
            # Ensure all expected keys exist
            required_keys = ["contact_info", "skills", "experience", "education", "certifications", "languages", "summary"]
            for key in required_keys:
                if key not in json_response:
                    if key == "contact_info":
                        json_response[key] = {"name": "Unknown", "email": "", "phone": "", "location": ""}
                    else:
                        json_response[key] = []
            
            # Extract candidate name for convenience
            json_response["name"] = json_response.get("contact_info", {}).get("name", "Unknown Candidate")
            
            return json_response
        except Exception as e:
            # Return basic structure if processing fails
            return {
                "error": f"Failed to process resume: {str(e)}",
                "name": "Unknown Candidate",
                "contact_info": {"name": "Unknown", "email": "", "phone": "", "location": ""},
                "skills": [],
                "experience": [],
                "education": [],
                "certifications": [],
                "languages": [],
                "summary": ""
            }
    
    def batch_process_resumes(self, resume_files: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple resumes in batch.
        
        Args:
            resume_files: Dictionary of file_name -> resume_text
            
        Returns:
            Dictionary of file_name -> processed_resume_data
        """
        results = {}
        
        for file_name, resume_text in resume_files.items():
            try:
                results[file_name] = self.process_resume(resume_text)
            except Exception as e:
                results[file_name] = {
                    "error": f"Failed to process {file_name}: {str(e)}",
                    "name": f"Unknown ({file_name})",
                    "contact_info": {"name": f"Unknown ({file_name})", "email": "", "phone": "", "location": ""},
                    "skills": [],
                    "experience": [],
                    "education": []
                }
        
        return results
    
    def extract_years_of_experience(self, resume_data: Dict[str, Any]) -> float:
        """
        Calculate total years of experience from resume data.
        
        Args:
            resume_data: Processed resume data
            
        Returns:
            Total years of experience as float
        """
        total_years = 0.0
        
        for exp in resume_data.get("experience", []):
            # Try to extract duration directly if available
            if "duration" in exp and exp["duration"]:
                # Parse phrases like "2 years", "2 years 3 months", etc.
                duration_text = exp["duration"].lower()
                years_match = re.search(r'(\d+\.?\d*)\s*years?', duration_text)
                months_match = re.search(r'(\d+\.?\d*)\s*months?', duration_text)
                
                years = float(years_match.group(1)) if years_match else 0
                months = float(months_match.group(1)) / 12.0 if months_match else 0
                
                total_years += years + months
                continue
            
            # If no duration, try to calculate from start/end dates
            start_date = exp.get("start_date", "")
            end_date = exp.get("end_date", "")
            
            # Skip if missing dates
            if not start_date:
                continue
            
            # Extract years from dates
            start_year_match = re.search(r'(\d{4})', start_date)
            
            if not start_year_match:
                continue
            
            start_year = int(start_year_match.group(1))
            
            # Handle current positions
            if not end_date or "present" in end_date.lower() or "current" in end_date.lower():
                from datetime import datetime
                end_year = datetime.now().year
                end_month = datetime.now().month
            else:
                end_year_match = re.search(r'(\d{4})', end_date)
                if not end_year_match:
                    continue
                end_year = int(end_year_match.group(1))
                
                # Try to extract month, default to middle of year if not found
                end_month_match = re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',
                                          end_date.lower())
                end_month = {
                    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
                }.get(end_month_match.group(1) if end_month_match else "", 6)
            
            # Extract start month, default to middle of year if not found
            start_month_match = re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',
                                       start_date.lower())
            start_month = {
                "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
            }.get(start_month_match.group(1) if start_month_match else "", 6)
            
            # Calculate years, including partial years for months
            experience_years = (end_year - start_year) + (end_month - start_month) / 12.0
            
            # Only count positive experiences (avoid errors in date parsing)
            if experience_years > 0:
                total_years += experience_years
        
        return total_years
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from text.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Parsed JSON as dictionary
        """
        # Find the first { and last } in the text
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            try:
                # Extract the JSON part
                json_text = text[start_idx:end_idx+1]
                return json.loads(json_text)
            except json.JSONDecodeError:
                # If parsing fails, try to find JSON in code blocks
                if "```json" in text:
                    parts = text.split("```json")
                    if len(parts) > 1:
                        json_part = parts[1].split("```")[0].strip()
                        try:
                            return json.loads(json_part)
                        except json.JSONDecodeError:
                            pass
                
                # If we still can't parse JSON, return empty dict
                return {}
        return {}

    def _analyze_text(self, prompt, temperature=0.7, max_tokens=1500):
        """
        Send a prompt to the Gemma model and get a response.
        
        Args:
            prompt: The text prompt to send to the model
            temperature: Controls randomness (lower is more deterministic)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            String response from the model
        """
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://localhost:8501",
                    "X-Title": "Resume Screener Application"
                },
                json={
                    "model": "google/gemma-3-12b-it:free",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                return response_data["choices"][0]["message"]["content"]
            else:
                raise ValueError("Invalid API response - missing choices")
                
        except Exception as e:
            import logging
            logging.error(f"Error calling OpenRouter API: {e}")
            return f"Error: {str(e)}"
