from crewai import Agent, Task
import requests
import json
from typing import Dict, Any
import re

class ResumeScoringAgent:
    """Agent responsible for scoring resumes against market requirements"""
    
    def __init__(self, openrouter_api_key: str):
        """
        Initialize the ResumeScoringAgent.
        
        Args:
            openrouter_api_key: OpenRouter API key for model access
        """
        # Use the API key provided in the parameter instead of loading from .env
        self.openrouter_api_key = openrouter_api_key
        
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key not provided")
        
        # Create the CrewAI agent
        self.agent = Agent(
            role="Resume Scoring Specialist",
            goal="Accurately evaluate resumes against current job market requirements",
            backstory=(
                "As an experienced talent evaluator with a background in technical "
                "recruitment, you excel at matching candidate profiles with job market "
                "requirements. You have helped thousands of job seekers understand how "
                "their skills align with what employers are looking for."
            ),
            verbose=True,
            allow_delegation=False,
            llm_config={"model": "google/gemma-3-12b-it:free"}  # Using free Gemma model
        )
    
    def score_resume(self, resume_data: Dict[str, Any], 
                market_data: Dict[str, Any],
                career_goal: str = None) -> Dict[str, Any]:
        """
        Score a resume against current market requirements.
        
        Args:
            resume_data: Resume data extracted by ResumeAnalyzerAgent
            market_data: Market trends data from JobMarketAnalyzerAgent
            career_goal: Optional career goal specified by the user
                
        Returns:
            Dictionary with resume scores and analysis
        """
        # Prepare the data for comparison
        resume_json = json.dumps(resume_data)
        market_json = json.dumps(market_data)
        
        # Include career goal in prompt if provided
        career_goal_text = f"Career Goal: {career_goal}\n\n" if career_goal else ""
        
        prompt = f"""
        You are an expert resume evaluator. Compare this resume against current market trends and provide a detailed scoring.
        
        {career_goal_text}Resume Data:
        {resume_json}
        
        Current Market Requirements:
        {market_json}
        
        Please analyze and provide:
        
        1. Overall match score (0-100)
        2. Skills match: Which skills from the resume match current in-demand skills
        3. Skills gaps: Which in-demand skills are missing from the resume
        4. Experience evaluation: How well does their experience align with market expectations
        5. Education evaluation: How well does their education align with market requirements
        6. Strengths: Top 3 strengths of this candidate based on the market
        7. Improvement areas: Top 3 areas to improve marketability
        
        IMPORTANT: Your response MUST be a JSON object with the following format:
        {{
            "Overall Match Score": 75,
            "Skills Match": {{
                "Matched Skills": ["Skill1", "Skill2", "Skill3"]
            }},
            "Skills Gaps": ["Missing Skill 1", "Missing Skill 2"],
            "Experience Evaluation": "Detailed evaluation here...",
            "Education Evaluation": "Detailed evaluation here...",
            "Strengths": ["Strength 1", "Strength 2", "Strength 3"],
            "Improvement Areas": ["Area 1", "Area 2", "Area 3"]
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
                    "model": "google/gemma-3-4b-it:free",  # Using free Gemma model
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are a resume evaluation expert who provides objective scoring based on market alignment. You MUST return ONLY valid JSON with the exact keys and format specified in the prompt. No preamble or explanations."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                }
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            # Print the entire response for debugging
            print("Full API response:", response_data)
            
            # Extract the response content
            if "choices" in response_data and len(response_data["choices"]) > 0:
                result = response_data["choices"][0]["message"]["content"]
                
                # Log the raw result for debugging
                print("Raw resume scoring API response:", result)
                
                # Process the response to extract JSON
                # First, try to find JSON-like content within code blocks
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
                        parsed_scoring = json.loads(json_content)
                        
                        # Ensure the response has the expected structure
                        expected_keys = [
                            "Overall Match Score", 
                            "Skills Match", 
                            "Skills Gaps", 
                            "Experience Evaluation", 
                            "Education Evaluation", 
                            "Strengths", 
                            "Improvement Areas"
                        ]
                        
                        # Check for missing keys and add default values
                        for key in expected_keys:
                            if key not in parsed_scoring:
                                if key == "Overall Match Score":
                                    parsed_scoring[key] = 50
                                elif key == "Skills Match":
                                    parsed_scoring[key] = {"Matched Skills": ["Unable to determine matched skills"]}
                                elif key == "Skills Gaps":
                                    parsed_scoring[key] = ["Unable to determine skill gaps"]
                                elif key in ["Experience Evaluation", "Education Evaluation"]:
                                    parsed_scoring[key] = "Unable to provide detailed evaluation"
                                elif key == "Strengths":
                                    parsed_scoring[key] = ["Resume was successfully processed", 
                                                        "Skills were identified", 
                                                        "Education history available"]
                                elif key == "Improvement Areas":
                                    parsed_scoring[key] = ["Consider adding more industry-specific keywords", 
                                                        "Quantify achievements with metrics", 
                                                        "Tailor resume to specific job roles"]
                        
                        return parsed_scoring
                    
                    else:
                        raise ValueError("Could not extract JSON content from the response")
                    
                except json.JSONDecodeError as json_err:
                    print(f"JSONDecodeError: {json_err}")
                    print("Content that failed to parse:", json_content)
                    raise
                    
            else:
                raise ValueError("Unexpected API response format: missing 'choices' field")
                    
        except Exception as e:
            print(f"Error in resume scoring: {str(e)}")
            
            # Return a fallback response with structured data
            return {
                "Overall Match Score": 60,
                "Skills Match": {
                    "Matched Skills": ["Unable to determine due to processing error"]
                },
                "Skills Gaps": ["Unable to determine due to processing error"],
                "Experience Evaluation": "Unable to evaluate due to processing error",
                "Education Evaluation": "Unable to evaluate due to processing error",
                "Strengths": ["Resume was successfully parsed",
                            "Information is available for review",
                            "Basic profile created"],
                "Improvement Areas": ["Try uploading your resume again",
                                    "Consider reformatting your resume for better analysis",
                                    "Check that your resume file is properly formatted"]
            }
    
    def create_scoring_task(self, resume_data: Dict[str, Any], 
                          market_data: Dict[str, Any]) -> Task:
        """
        Create a CrewAI Task for resume scoring.
        
        Args:
            resume_data: Resume data extracted by ResumeAnalyzerAgent
            market_data: Market trends data from JobMarketAnalyzerAgent
            
        Returns:
            CrewAI Task object
        """
        return Task(
            description="Evaluate this resume against current market requirements and provide a detailed scoring analysis",
            expected_output="Structured JSON with resume scores, skill matches, gaps, and improvement recommendations",
            agent=self.agent,
            context={
                "resume_data": resume_data,
                "market_data": market_data
            }
        )
