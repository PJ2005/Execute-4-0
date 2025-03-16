import re
from crewai import Agent
from typing import Dict, Any, List
import logging
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

class JobAnalyzerAgent:
    """Agent responsible for analyzing job descriptions and extracting key requirements."""
    
    def __init__(self, openrouter_api_key=None):
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key not found. Please set OPENROUTER_API_KEY in .env file")
            
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the CrewAI agent for job analysis."""
        return Agent(
            role="Job Analysis Specialist",
            goal="Extract key requirements and skills from job descriptions",
            backstory=(
                "You are an expert in parsing job descriptions and identifying key "
                "requirements, skills, and qualifications. Your analysis helps match "
                "the right candidates to the right positions."
            ),
            verbose=True,
            allow_delegation=False,
            llm_config={"model": "google/gemma-3-12b-it:free"}
        )
    
    def analyze_job_description(self, job_description: str) -> Dict[str, Any]:
        """
        Extract structured requirements from a job description.
        
        Args:
            job_description: The job description text
            
        Returns:
            Dictionary with structured job requirements
        """
        from utils.json_handler import JsonHandler
        
        prompt = f"""
        Extract the following information from this job description:
        - Job Title
        - Industry
        - Years of Experience Required
        - Education Requirements
        - Seniority Level
        - Required Skills (technical and soft skills explicitly mentioned as required)
        - Preferred Skills (skills mentioned as nice-to-have or preferred)
        - Key Responsibilities

        Format your response as a valid JSON object with the following structure exactly:
        {{
          "job_title": "Title of the position",
          "industry": "Industry sector",
          "years_of_experience": "Experience requirement (e.g., 3-5 years)",
          "education_requirements": "Required education level",
          "seniority_level": "Junior/Mid-level/Senior/etc.",
          "required_skills": ["Skill 1", "Skill 2", ...],
          "preferred_skills": ["Skill 1", "Skill 2", ...],
          "key_responsibilities": ["Responsibility 1", "Responsibility 2", ...]
        }}

        JOB DESCRIPTION:
        {job_description}
        
        Return ONLY valid JSON.
        """
        
        try:
            response_text = self._analyze_text(prompt, temperature=0.2)
            structured_data = JsonHandler.extract_json(response_text)
            
            if not structured_data:
                structured_data = {}
                
            # Ensure all keys exist
            required_keys = [
                "job_title", "industry", "years_of_experience", 
                "education_requirements", "seniority_level", 
                "required_skills", "preferred_skills", "key_responsibilities"
            ]
            
            for key in required_keys:
                if key not in structured_data:
                    if key.endswith("_skills") or key == "key_responsibilities":
                        structured_data[key] = ["None specified"]
                    else:
                        structured_data[key] = "Not specified"
                        
            # Make sure skills and responsibilities are always lists
            for key in ["required_skills", "preferred_skills", "key_responsibilities"]:
                if not structured_data.get(key) or not isinstance(structured_data[key], list):
                    structured_data[key] = ["None specified"]
                    
            # Make sure empty lists get a default value
            for key in ["required_skills", "preferred_skills", "key_responsibilities"]:
                if not structured_data.get(key) or len(structured_data[key]) == 0:
                    structured_data[key] = ["None specified"]
            
            # Add extracted critical keywords
            structured_data["critical_keywords"] = self.extract_critical_keywords(structured_data)
            
            return JsonHandler.clean_none_values(structured_data)
            
        except Exception as e:
            logging.error(f"Error analyzing job description: {e}")
            return {
                "job_title": "Data Analysis Position",  # Default to generic title
                "industry": "Technology",
                "years_of_experience": "3-5 years",
                "education_requirements": "Bachelor's degree",
                "seniority_level": "Mid-level",
                "required_skills": ["Communication", "Problem Solving"],
                "preferred_skills": ["Project Management"],
                "key_responsibilities": ["Data Analysis", "Reporting"],
                "critical_keywords": ["analysis", "data", "communication"],
                "error": f"Error details: {str(e)}"
            }
    
    def extract_critical_keywords(self, job_analysis: Dict[str, Any]) -> List[str]:
        """
        Extract the most critical keywords from job analysis.
        
        Args:
            job_analysis: The structured job analysis
            
        Returns:
            List of critical keywords
        """
        keywords = set()
        
        # Add all required skills
        for skill in job_analysis.get("required_skills", []):
            keywords.add(skill.lower())
            
        # Add all preferred skills
        for skill in job_analysis.get("preferred_skills", []):
            keywords.add(skill.lower())
            
        # Extract key phrases from job title
        if "job_title" in job_analysis:
            title = job_analysis["job_title"].lower()
            title_words = title.split()
            for word in title_words:
                if len(word) > 3 and word not in ["and", "the", "for"]:
                    keywords.add(word)
                    
        # Extract key phrases from responsibilities
        for resp in job_analysis.get("key_responsibilities", []):
            resp = resp.lower()
            
            # Extract key technical terms
            tech_terms = re.findall(r'\b[a-z0-9]+([\+\#])?(\.[a-z0-9]+)+\b', resp)
            for term in tech_terms:
                if term and len(term) > 0:
                    keywords.add(term[0])
                
            # Extract key phrases from responsibilities
            words = resp.lower().split()
            for word in words:
                if len(word) > 4 and word not in ['with', 'that', 'this', 'and', 'the', 'for']:
                    keywords.add(word)
        
        return list(keywords)
    
    def generate_interview_questions(self, job_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate interview questions based on job requirements.
        
        Args:
            job_analysis: The structured job analysis
            
        Returns:
            List of interview questions
        """
        required_skills = ", ".join(job_analysis.get("required_skills", []))
        responsibilities = ", ".join(job_analysis.get("key_responsibilities", []))
        job_title = job_analysis.get("job_title", "the position")
        
        prompt = f"""
        Generate 5 technical interview questions for {job_title} based on these requirements:
        
        Required Skills: {required_skills}
        
        Key Responsibilities: {responsibilities}
        
        Create questions that assess technical competency, problem-solving abilities, and experience.
        Each question should be specific, challenging, and reveal the candidate's expertise level.
        Format as a simple numbered list.
        """
        
        try:
            response = self._analyze_text(prompt, temperature=0.7)
            
            # Extract questions from numbered list
            questions = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('- ')):
                    # Remove number/bullet and any trailing punctuation
                    clean_line = line.split('. ', 1)[-1] if '. ' in line else line
                    clean_line = clean_line[2:] if clean_line.startswith('- ') else clean_line
                    questions.append(clean_line)
            
            return questions if questions else ["Can you describe your experience with the technologies mentioned in the job description?", 
                                             "How would you approach the key responsibilities of this role?",
                                             "Tell me about a challenging project you've worked on that's relevant to this position.",
                                             "How do you stay updated with the latest trends in this field?",
                                             "What questions do you have about the role or company?"]
        except Exception as e:
            return [f"Error generating interview questions: {str(e)}"]
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON object from text.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Parsed JSON object
        """
        try:
            # First try direct JSON parsing
            return json.loads(text)
        except:
            # Try to extract JSON from text
            json_start = text.find('{')
            json_end = text.rfind('}')
            
            if json_start >= 0 and json_end >= 0:
                try:
                    json_str = text[json_start:json_end+1]
                    return json.loads(json_str)
                except:
                    # Try to find JSON in code blocks
                    if "```json" in text:
                        json_blocks = text.split("```json")
                        if len(json_blocks) > 1:
                            json_content = json_blocks[1].split("```")[0]
                            try:
                                return json.loads(json_content)
                            except:
                                pass
                    elif "```" in text:
                        json_blocks = text.split("```")
                        if len(json_blocks) > 1:
                            json_content = json_blocks[1].strip()
                            try:
                                return json.loads(json_content)
                            except:
                                pass
            
            # If all attempts failed, return default structure
            return {
                "job_title": "Position",
                "industry": "Not specified",
                "years_of_experience": "Not specified",
                "education_requirements": "Not specified",
                "seniority_level": "Not specified",
                "required_skills": [],
                "preferred_skills": [],
                "key_responsibilities": []
            }

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
                    "messages": [
                        {"role": "system", "content": "You are an expert job analyzer who extracts structured information from job descriptions. Always return valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
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
            logging.error(f"Error calling OpenRouter API: {e}")
            return f"Error: {str(e)}"
