from crewai import Agent, Task
import requests
from typing import Dict, Any, List
from dotenv import load_dotenv
import os
import json

class ResumeAnalyzerAgent:
    """Agent responsible for analyzing resumes and extracting key information"""
    
    def __init__(self, openrouter_api_key: str):
        """
        Initialize the ResumeAnalyzerAgent with OpenRouter API key.
        
        Args:
            openrouter_api_key: OpenRouter API key for model access
        """
        self.openrouter_api_key = openrouter_api_key
        
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key not provided")
        
        # Create the CrewAI agent
        self.agent = Agent(
            role="Resume Analyzer Expert",
            goal="Extract key information from resumes with high accuracy",
            backstory=(
                "As an expert in resume analysis with years of experience in HR and "
                "recruitment, you have developed a keen eye for identifying skills, "
                "qualifications, and experience from resume documents. You specialize "
                "in parsing unstructured resume data and organizing it into useful categories."
            ),
            verbose=True,
            allow_delegation=False,
            llm_config={"model": "google/gemma-3-12b-it:free"}
        )
    
    def extract_resume_information(self, resume_text: str, career_goal: str = None) -> Dict[str, Any]:
        """
        Extract structured information from a resume using OpenRouter API.
        
        Args:
            resume_text: Plain text extracted from a resume PDF
            career_goal: Optional career goal specified by the user
            
        Returns:
            Dictionary containing structured resume information
        """
        # Include career goal in the prompt if provided
        career_goal_text = ""
        if (career_goal):
            career_goal_text = f"""
            The user has specified their career goal as: "{career_goal}"
            
            Keep this career goal in mind when extracting information from the resume.
            Highlight skills, experiences and education relevant to this career aspiration.
            """
            
        prompt = f"""
        Analyze the following resume text and extract key information in JSON format.
        Focus on these categories:
        1. Personal Information (name, email, phone, location)
        2. Skills (technical skills, soft skills)
        3. Education (degrees, institutions, dates, GPAs if available)
        4. Work Experience (job titles, companies, dates, key achievements)
        5. Projects (if any)
        6. Certifications (if any)
        
        {career_goal_text}
        
        Resume Text:
        {resume_text}
        
        You MUST return ONLY a valid JSON object with the following format:
        {{
            "personalInformation": {{
                "name": "Full Name",
                "email": "email@example.com",
                "phone": "123-456-7890",
                "location": "City, State"
            }},
            "skills": ["Skill 1", "Skill 2", "Skill 3"],
            "education": [
                {{
                    "degree": "Degree Name",
                    "institution": "University Name",
                    "startDate": "Start Date",
                    "endDate": "End Date",
                    "gpa": "GPA if available"
                }}
            ],
            "workExperience": [
                {{
                    "jobTitle": "Job Title",
                    "company": "Company Name",
                    "startDate": "Start Date",
                    "endDate": "End Date",
                    "description": "Job description and achievements"
                }}
            ],
            "projects": [
                {{
                    "name": "Project Name",
                    "description": "Project description"
                }}
            ],
            "certifications": [
                {{
                    "name": "Certification Name",
                    "issuer": "Issuing Organization",
                    "date": "Issue Date"
                }}
            ]
        }}
        
        Only include sections that are present in the resume. Make sure to provide ONLY the JSON object with no additional text or explanations.
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
                        {"role": "system", "content": "You are a resume parsing expert that extracts structured data from resume texts. Output ONLY valid JSON."},
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
                print("Raw resume extraction response:", result)
                
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
                        parsed_info = json.loads(json_content)
                        
                        # Ensure at least some basic sections exist
                        if "personalInformation" not in parsed_info:
                            parsed_info["personalInformation"] = {"name": "Not Found"}
                        
                        if "skills" not in parsed_info:
                            parsed_info["skills"] = []
                            
                        if "workExperience" not in parsed_info:
                            parsed_info["workExperience"] = []
                            
                        if "education" not in parsed_info:
                            parsed_info["education"] = []
                        
                        return parsed_info
                    
                    else:
                        raise ValueError("Could not extract JSON content from the response")
                    
                except json.JSONDecodeError as json_err:
                    print(f"JSONDecodeError: {json_err}")
                    print("Content that failed to parse:", json_content)
                    raise
                    
            else:
                raise ValueError("Unexpected API response format: missing 'choices' field")
                    
        except Exception as e:
            print(f"Error in resume analysis: {str(e)}")
            
            # Return a minimal fallback structure
            return {
                "personalInformation": {
                    "name": "Not available",
                    "email": "",
                    "phone": "",
                    "location": ""
                },
                "skills": ["Unable to extract skills"],
                "education": [],
                "workExperience": [],
                "projects": [],
                "certifications": [],
                "error": str(e)
            }
    
    def create_analysis_task(self, resume_text: str) -> Task:
        """
        Create a CrewAI Task for resume analysis.
        
        Args:
            resume_text: Plain text extracted from a resume PDF
            
        Returns:
            CrewAI Task object
        """
        return Task(
            description=f"Analyze this resume and extract key information: {resume_text[:500]}...",
            expected_output="Structured JSON with personal details, skills, education, and experience",
            agent=self.agent
        )
