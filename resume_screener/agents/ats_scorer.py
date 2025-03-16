import re
from crewai import Agent
from typing import Dict, Any, List
import logging
import os
import requests
import json
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

class ATSScoringAgent:
    """Agent responsible for implementing ATS scoring algorithms."""
    
    def __init__(self, openrouter_api_key=None):
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key not found. Please set OPENROUTER_API_KEY in .env file")
        
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the CrewAI agent for resume scoring."""
        return Agent(
            role="ATS Scoring Specialist",
            goal="Accurately score resumes against job requirements",
            backstory=(
                "You are an expert in applicant tracking systems and resume evaluation. "
                "You specialize in analyzing how well resumes match job requirements "
                "and providing objective scoring based on industry-standard ATS algorithms."
            ),
            verbose=True,
            allow_delegation=False,
            llm_config={"model": "google/gemma-3-12b-it:free"}
        )
    
    def score_resume(self, job_requirements: Dict[str, Any], resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a resume against job requirements using ATS algorithms.
        
        Args:
            job_requirements: The job requirements dictionary
            resume_data: The processed resume data
            
        Returns:
            Dictionary containing ATS scores
        """
        try:
            # Extract key information from job requirements and resume
            job_title = job_requirements.get("job_title", "")
            required_skills = set(skill.lower() for skill in job_requirements.get("required_skills", []))
            preferred_skills = set(skill.lower() for skill in job_requirements.get("preferred_skills", []))
            all_required = required_skills.union(preferred_skills)
            
            candidate_skills = set(skill.lower() for skill in resume_data.get("skills", []))
            experience = resume_data.get("experience", [])
            if not experience and "workExperience" in resume_data:
                # Handle alternative key from resume_career_ai_app format
                experience = resume_data.get("workExperience", [])
                
            education = resume_data.get("education", [])
            
            # Calculate skill match
            required_match = self._calculate_skill_match(required_skills, candidate_skills)
            preferred_match = self._calculate_skill_match(preferred_skills, candidate_skills)
            
            # Weight required skills higher than preferred
            skill_match_score = (required_match * 0.7) + (preferred_match * 0.3)
            
            # Calculate experience match
            experience_match_score = self._calculate_experience_match(job_requirements, resume_data)
            
            # Calculate education match
            education_match_score = self._calculate_education_match(job_requirements, resume_data)
            
            # Calculate keyword density score
            keyword_score = self._calculate_keyword_density(job_requirements, resume_data)
            
            # Calculate title match score
            title_match_score = self._calculate_title_match(job_title, experience)
            
            # Generate final ATS score - weighted average
            ats_score = (
                skill_match_score * 0.40 +
                experience_match_score * 0.30 +
                education_match_score * 0.15 +
                keyword_score * 0.10 +
                title_match_score * 0.05
            )
            
            # Use GPT for deeper analysis to extract strengths and weaknesses
            detailed_analysis_result = self._get_comprehensive_analysis(job_requirements, resume_data)
            
            # Compile detailed results
            result = {
                "ats_score": min(ats_score, 100.0),  # Cap at 100%
                "skill_match": skill_match_score,
                "experience_match": experience_match_score,
                "education_match": education_match_score,
                "keyword_density": keyword_score,
                "title_match": title_match_score,
                "matching_skills": list(candidate_skills.intersection(all_required)),
                "missing_skills": list(all_required - candidate_skills),
                "detailed_analysis": detailed_analysis_result.get("analysis", ""),
                "strengths": detailed_analysis_result.get("strengths", []),
                "improvement_areas": detailed_analysis_result.get("improvement_areas", [])
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error scoring resume: {e}")
            return {
                "error": f"Failed to score resume: {str(e)}",
                "ats_score": 50.0,  # Default to middle score
                "skill_match": 50.0,
                "experience_match": 50.0,
                "education_match": 50.0,
                "keyword_density": 50.0,
                "title_match": 50.0,
                "matching_skills": [],
                "missing_skills": [],
                "detailed_analysis": "Error analyzing resume"
            }
    
    def _calculate_skill_match(self, required_skills: set, candidate_skills: set) -> float:
        """
        Calculate skill match percentage.
        
        Args:
            required_skills: Set of required skills (lowercase)
            candidate_skills: Set of candidate skills (lowercase)
            
        Returns:
            Match percentage (0.0-100.0)
        """
        if not required_skills:
            return 100.0  # No required skills means perfect match
            
        # Count matches
        matches = 0
        for req_skill in required_skills:
            # Check for exact match
            if req_skill in candidate_skills:
                matches += 1
                continue
                
            # Check for partial matches
            for cand_skill in candidate_skills:
                if (req_skill in cand_skill) or (cand_skill in req_skill):
                    matches += 0.5  # Partial credit for partial matches
                    break
        
        # Calculate percentage
        return min((matches / len(required_skills)) * 100.0, 100.0)
    
    def _calculate_experience_match(self, job_requirements: Dict[str, Any], resume_data: Dict[str, Any]) -> float:
        """
        Calculate how well candidate's experience matches job requirements.
        
        Args:
            job_requirements: Job requirements dictionary
            resume_data: Resume data dictionary
            
        Returns:
            Match percentage (0.0-100.0)
        """
        # Extract required experience
        req_exp_str = job_requirements.get("years_of_experience", "Not specified")
        
        # Parse required years (e.g., "3-5 years" -> 4.0)
        required_years = 0.0
        if isinstance(req_exp_str, str):
            years_match = re.search(r'(\d+\.?\d*)\s*(?:-|to)\s*(\d+\.?\d*)', req_exp_str)
            if years_match:
                # Range specified, use midpoint
                min_years = float(years_match.group(1))
                max_years = float(years_match.group(2))
                required_years = (min_years + max_years) / 2.0
            else:
                # Single value
                single_match = re.search(r'(\d+\.?\d*)', req_exp_str)
                if single_match:
                    required_years = float(single_match.group(1))
        
        # If no experience required or couldn't parse
        if required_years <= 0:
            required_years = 2.0  # Default to 2 years if not specified
        
        # Calculate total years from resume
        candidate_years = self._calculate_total_experience_years(resume_data.get("experience", []))
        
        # Calculate match percentage
        if candidate_years >= required_years:
            return 100.0
        elif candidate_years <= 0:
            return 0.0
        else:
            # Partial match based on proportion
            return (candidate_years / required_years) * 100.0
    
    def _calculate_total_experience_years(self, experience: List[Dict[str, Any]]) -> float:
        """
        Calculate total years of experience from experience entries.
        
        Args:
            experience: List of experience dictionaries
            
        Returns:
            Total years of experience as float
        """
        total_years = 0.0
        
        for exp in experience:
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
    
    def _calculate_education_match(self, job_requirements: Dict[str, Any], resume_data: Dict[str, Any]) -> float:
        """
        Calculate education match percentage.
        
        Args:
            job_requirements: Job requirements dictionary
            resume_data: Resume data dictionary
            
        Returns:
            Match percentage (0.0-100.0)
        """
        education_req = job_requirements.get("education_requirements", "").lower()
        candidate_education = resume_data.get("education", [])
        
        # Define education levels and their scores
        education_levels = {
            "high school": 1,
            "associate": 2,
            "diploma": 2,
            "bachelor": 3,
            "bs": 3,
            "ba": 3,
            "undergraduate": 3,
            "master": 4,
            "mba": 4,
            "graduate": 4,
            "phd": 5,
            "doctorate": 5,
        }
        
        # Determine required education level
        required_level = 0
        for level, score in education_levels.items():
            if level in education_req:
                required_level = max(required_level, score)
        
        # Default to bachelor's if requirement is unclear
        if required_level == 0:
            required_level = 3
        
        # Find candidate's highest education level
        candidate_level = 0
        for edu in candidate_education:
            degree = edu.get("degree", "").lower()
            for level, score in education_levels.items():
                if level in degree:
                    candidate_level = max(candidate_level, score)
        
        # Score based on education level
        if candidate_level >= required_level:
            return 100.0
        elif candidate_level == 0:
            return 0.0
        else:
            # Partial credit for some education
            return (candidate_level / required_level) * 100.0
    
    def _calculate_keyword_density(self, job_requirements: Dict[str, Any], resume_data: Dict[str, Any]) -> float:
        """
        Calculate keyword density match.
        
        Args:
            job_requirements: Job requirements dictionary
            resume_data: Resume data dictionary
            
        Returns:
            Match percentage (0.0-100.0)
        """
        # Extract keywords from job requirements
        keywords = set()
        for skill in job_requirements.get("required_skills", []):
            keywords.add(skill.lower())
        
        for skill in job_requirements.get("preferred_skills", []):
            keywords.add(skill.lower())
            
        # Add critical keywords extracted from responsibilities
        for keyword in job_requirements.get("critical_keywords", []):
            keywords.add(keyword.lower())
        
        # Filter out common words
        common_words = {'and', 'the', 'to', 'of', 'for', 'in', 'a', 'with', 'on', 'an', 'this', 'that', 'be', 'as'}
        keywords = {k for k in keywords if k not in common_words and len(k) > 2}
        
        if not keywords:
            return 100.0  # No keywords to match
        
        # Create resume text corpus
        resume_text = ""
        resume_text += " ".join(resume_data.get("skills", []))
        resume_text += " " + resume_data.get("summary", "")
        
        for exp in resume_data.get("experience", []):
            resume_text += " " + exp.get("title", "")
            resume_text += " " + exp.get("company", "")
            resume_text += " " + exp.get("description", "")
        
        resume_text = resume_text.lower()
        
        # Count matches
        matches = 0
        for keyword in keywords:
            if keyword in resume_text:
                matches += 1
        
        return (matches / len(keywords)) * 100.0
    
    def _calculate_title_match(self, job_title: str, experience: List[Dict[str, Any]]) -> float:
        """
        Calculate match between job title and candidate's most recent job titles.
        
        Args:
            job_title: Job title
            experience: List of experience dictionaries
            
        Returns:
            Match percentage (0.0-100.0)
        """
        if not job_title or not experience:
            return 50.0  # Neutral score if no data
            
        # Normalize job title
        job_title = job_title.lower()
        job_title_words = set(job_title.split())
        
        # Look at the most recent 3 jobs
        best_match = 0.0
        for exp in experience[:3]:
            candidate_title = exp.get("title", "").lower()
            candidate_title_words = set(candidate_title.split())
            
            # Calculate Jaccard similarity
            intersection = job_title_words.intersection(candidate_title_words)
            union = job_title_words.union(candidate_title_words)
            
            if not union:
                continue
                
            similarity = (len(intersection) / len(union)) * 100.0
            best_match = max(best_match, similarity)
        
        return best_match
    
    def _get_comprehensive_analysis(self, job_requirements: Dict[str, Any], resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive analysis of resume against job requirements using LLM.
        
        Args:
            job_requirements: Job requirements dictionary
            resume_data: Resume data dictionary
            
        Returns:
            Dictionary with analysis, strengths, and improvement areas
        """
        from utils.json_handler import JsonHandler
        
        # Ensure all values are strings to prevent None concatenation issues
        prompt = f"""
        Analyze how well this candidate matches the job requirements. Provide:
        1. A concise analysis (3-4 sentences) of the overall fit
        2. A list of 3-5 key strengths of the candidate for this role
        3. A list of 2-4 areas for improvement or gaps in the candidate's profile
        
        JOB REQUIREMENTS:
        - Title: {job_requirements.get('job_title', 'Not specified') or 'Not specified'}
        - Required Skills: {', '.join(job_requirements.get('required_skills', []) or ['Not specified'])}
        - Preferred Skills: {', '.join(job_requirements.get('preferred_skills', []) or ['Not specified'])}
        - Experience: {job_requirements.get('years_of_experience', 'Not specified') or 'Not specified'}
        - Education: {job_requirements.get('education_requirements', 'Not specified') or 'Not specified'}
        - Level: {job_requirements.get('seniority_level', 'Not specified') or 'Not specified'}
        - Industry: {job_requirements.get('industry', 'Not specified') or 'Not specified'}
        - Responsibilities: {', '.join((job_requirements.get('key_responsibilities', []) or [])[:5])[:300]}
        
        CANDIDATE PROFILE:
        - Name: {resume_data.get('name', 'Not specified') or 'Not specified'}
        - Skills: {', '.join(resume_data.get('skills', []) or ['Not specified'])}
        """
        
        # Extract experience count safely
        experience_list = resume_data.get('experience', resume_data.get('workExperience', []) or [])
        experience_count = len(experience_list) if experience_list else 0
        
        # Extract most recent position safely
        most_recent_role = "Not specified"
        if experience_count > 0:
            most_recent = experience_list[0]
            if isinstance(most_recent, dict):
                most_recent_role = most_recent.get('title', most_recent.get('jobTitle', 'Not specified')) or 'Not specified'
        
        # Extract education safely
        education_list = resume_data.get('education', []) or []
        education_info = "Not specified"
        if education_list and len(education_list) > 0:
            most_recent_edu = education_list[0]
            if isinstance(most_recent_edu, dict):
                education_info = most_recent_edu.get('degree', 'Not specified') or 'Not specified'
                
        # Continue building the prompt
        prompt += f"""
        - Experience: {experience_count} positions
        - Most recent role: {most_recent_role}
        - Education: {education_info}
        """
        
        # Get summary if available
        summary = resume_data.get('summary', '') or ''
        if summary:
            prompt += f"\n- Summary: {summary[:200]}"
        
        # Add output format instructions
        prompt += """
        
        Format your response as a JSON object with the following keys:
        {
          "analysis": "Overall analysis of fit for the role",
          "strengths": ["Strength 1", "Strength 2", "Strength 3"],
          "improvement_areas": ["Area 1", "Area 2", "Area 3"]
        }
        
        Return ONLY valid JSON.
        """
        
        try:
            response_text = self._analyze_text(prompt, temperature=0.5)
            
            # Use the JsonHandler to safely extract JSON
            json_data = JsonHandler.extract_json(response_text)
            
            if json_data:
                return json_data
            else:
                # Fallback structure
                return {
                    "analysis": self._generate_detailed_analysis(job_requirements, resume_data),
                    "strengths": ["Candidate has relevant experience", "Candidate has some matching skills"],
                    "improvement_areas": ["Consider adding more industry-specific keywords", "Tailor resume to highlight relevant experience"]
                }
                
        except Exception as e:
            logging.error(f"Error generating comprehensive analysis: {e}")
            return {
                "analysis": f"Error generating analysis: {str(e)}",
                "strengths": [],
                "improvement_areas": []
            }
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON object from text."""
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
            
            # If all attempts failed, return empty dict
            return {}
                    
    def _generate_detailed_analysis(self, job_requirements: Dict[str, Any], resume_data: Dict[str, Any]) -> str:
        """
        Generate detailed analysis of resume vs job requirements using Gemma.
        
        Args:
            job_requirements: Job requirements dictionary
            resume_data: Resume data dictionary
            
        Returns:
            Detailed analysis text
        """
        # Ensure all fields are strings to avoid None concatenation issues
        prompt = f"""
        Analyze how well this candidate matches the job requirements. Provide a concise 
        analysis of strengths and weaknesses in terms of skills, experience, and education.
        
        JOB REQUIREMENTS:
        - Title: {job_requirements.get('job_title', 'Not specified')}
        - Required Skills: {', '.join(job_requirements.get('required_skills', []) or ['Not specified'])}
        - Preferred Skills: {', '.join(job_requirements.get('preferred_skills', []) or ['Not specified'])}
        - Experience: {job_requirements.get('years_of_experience', 'Not specified') or 'Not specified'}
        - Education: {job_requirements.get('education_requirements', 'Not specified') or 'Not specified'}
        - Level: {job_requirements.get('seniority_level', 'Not specified') or 'Not specified'}
        
        CANDIDATE PROFILE:
        - Skills: {', '.join(resume_data.get('skills', []) or ['Not specified'])}
        - Experience: {len(resume_data.get('experience', resume_data.get('workExperience', [])) or [])} positions
        """
        
        # Extract most recent position safely
        experience_list = resume_data.get('experience', resume_data.get('workExperience', []) or [])
        most_recent_role = "Not specified"
        
        if experience_list and len(experience_list) > 0:
            most_recent = experience_list[0]
            if isinstance(most_recent, dict):
                most_recent_role = most_recent.get('title', most_recent.get('jobTitle', 'Not specified')) or 'Not specified'
        
        # Extract education safely
        education_list = resume_data.get('education', []) or []
        education_info = "Not specified"
        
        if education_list and len(education_list) > 0:
            most_recent_edu = education_list[0]
            if isinstance(most_recent_edu, dict):
                education_info = most_recent_edu.get('degree', 'Not specified') or 'Not specified'
                
        # Continue building the prompt
        prompt += f"""
        - Most recent role: {most_recent_role}
        - Education: {education_info}
        - Summary: {(resume_data.get('summary', '') or '')[:200]}
        """
        
        # Use the safer JsonHandler utility for the response
        try:
            response_text = self._analyze_text(prompt, temperature=0.5)
            return response_text
        except Exception as e:
            logging.error(f"Error generating detailed analysis: {e}")
            return f"Error generating analysis: {str(e)}"
    
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
                        {"role": "system", "content": "You are an expert ATS system that evaluates resumes against job requirements with high precision."},
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

