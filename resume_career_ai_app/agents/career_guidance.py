from crewai import Agent, Task
import requests
from typing import Dict, Any, List
import json
from dotenv import load_dotenv
import os

class CareerGuidanceAgent:
    """Agent responsible for providing personalized career guidance"""
    
    def __init__(self, openrouter_api_key: str):
        """
        Initialize the CareerGuidanceAgent.
        
        Args:
            openrouter_api_key: OpenRouter API key for model access
        """
        # Instead, directly use the API key passed to the constructor
        self.openrouter_api_key = openrouter_api_key
    
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key not provided")
        
        # Create the CrewAI agent
        self.agent = Agent(
            role="Career Guidance Counselor",
            goal="Provide actionable and personalized career advice",
            backstory=(
                "As a seasoned career counselor with 15+ years of experience, you've "
                "guided thousands of professionals through career transitions and growth. "
                "You excel at understanding an individual's unique strengths and aspirations, "
                "then mapping these to practical career paths and development strategies."
            ),
            verbose=True,
            allow_delegation=False,
            llm_config={"model": "google/gemma-3-12b-it:free"}
        )
    
    def generate_career_recommendations(self, resume_data: Dict[str, Any], 
                                  market_data: Dict[str, Any],
                                  scoring_data: Dict[str, Any],
                                  career_goal: str = None) -> Dict[str, Any]:
        """
        Generate personalized career recommendations.
        
        Args:
            resume_data: Resume data extracted by ResumeAnalyzerAgent
            market_data: Market trends data from JobMarketAnalyzerAgent
            scoring_data: Resume scoring data from ResumeScoringAgent
            career_goal: Optional career goal specified by the user
            
        Returns:
            Dictionary with career recommendations and guidance
        """
        # Prepare data for the prompt
        resume_skills = json.dumps(resume_data.get("skills", []))
        resume_experience = json.dumps(resume_data.get("workExperience", []))
        market_trends = json.dumps(market_data)
        resume_score = json.dumps(scoring_data)
        
        # Include career goal if provided
        career_goal_text = f"\nCareer Goal: {career_goal}" if career_goal else ""
        
        prompt = f"""
        Based on the following information, provide personalized career guidance in JSON format:{career_goal_text}
        
        Resume Skills: {resume_skills}
        
        Work Experience: {resume_experience}
        
        Current Market Trends: {market_trends}
        
        Resume Scoring: {resume_score}
        
        Please generate a structured JSON response with EXACTLY these top-level keys:
        1. "overallAssessment" - Brief overall assessment of the candidate's profile (string)
        2. "careerPathRecommendations" - Array of objects, each with "Rank" (number), "Career Path" (string), and "Reasoning" (string)
        3. "skillDevelopmentPlan" - Array of objects, each with "Skill" (string), "Description" (string), and "Resources" (array of strings)
        4. "shortTermActions" - Array of strings with immediate actions (1-3 months)
        5. "mediumTermStrategy" - Array of strings with medium-term strategy (1-2 years)
        6. "longTermVision" - Array of strings with long-term vision (5+ years)
        
        IMPORTANT: Return ONLY valid JSON with these exact keys. Use double quotes for all keys and string values.
        Do not add any explanatory text before or after the JSON.
        """
        
        try:
            # Use a free model from OpenRouter - same as what's working in other functions
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    # Using same model as other functions that are working
                    "model": "google/gemma-3-4b-it:free",
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are a career guidance expert who provides structured JSON recommendations. Always return valid JSON with double quotes for all keys and string values."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2
                }
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            # Print full response for debugging
            print("Full API response for career guidance:", response_data)
            
            # Extract the content
            if "choices" in response_data and len(response_data["choices"]) > 0:
                result = response_data["choices"][0]["message"]["content"]
                
                # Log the raw result for debugging
                print("Raw career guidance API response:", result)
                
                # Process the response to ensure it's proper JSON
                try:
                    # Remove markdown code block formatting if present
                    cleaned_result = result
                    
                    # Check for code blocks
                    if "```json" in cleaned_result:
                        cleaned_result = cleaned_result.split("```json")[1]
                        
                    if "```" in cleaned_result:
                        parts = cleaned_result.split("```")
                        for part in parts:
                            if "{" in part and "}" in part:
                                cleaned_result = part
                                break
                    
                    # Remove any non-JSON text before and after
                    start_idx = cleaned_result.find('{')
                    end_idx = cleaned_result.rfind('}') + 1
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        json_content = cleaned_result[start_idx:end_idx]
                        
                        # Try to parse the JSON
                        parsed_recommendations = json.loads(json_content)
                        
                        # If the JSON is nested inside a "Career Guidance" key, flatten it
                        if "Career Guidance" in parsed_recommendations:
                            inner_content = parsed_recommendations["Career Guidance"]
                            if isinstance(inner_content, dict):
                                # Handle numerically prefixed keys
                                new_dict = {}
                                for key, value in inner_content.items():
                                    # Remove numeric prefixes like "1. " from keys
                                    if '. ' in key and key[0].isdigit():
                                        new_key = key.split('. ', 1)[1].lower().replace(' ', '')
                                        new_dict[new_key] = value
                                    else:
                                        # Convert keys to camelCase
                                        new_key = key.lower().replace(' ', '')
                                        new_dict[new_key] = value
                                
                                return new_dict
                            else:
                                return inner_content
                        else:
                            # Normalize keys to camelCase if needed
                            normalized = {}
                            key_mapping = {
                                "overall assessment": "overallAssessment",
                                "career path recommendations": "careerPathRecommendations",
                                "skill development plan": "skillDevelopmentPlan",
                                "short term actions": "shortTermActions", 
                                "medium term strategy": "mediumTermStrategy",
                                "long term vision": "longTermVision"
                            }
                            
                            for key, value in parsed_recommendations.items():
                                # Check if key needs normalization
                                normalized_key = key
                                for old_key, new_key in key_mapping.items():
                                    if key.lower() == old_key or key.lower().replace('-', ' ') == old_key:
                                        normalized_key = new_key
                                        break
                                
                                # Handle numbered keys
                                if '. ' in key and key[0].isdigit():
                                    base_key = key.split('. ', 1)[1].lower().replace(' ', '')
                                    for old_key, new_key in key_mapping.items():
                                        if base_key == old_key or base_key.replace('-', ' ') == old_key:
                                            normalized_key = new_key
                                            break
                                
                                normalized[normalized_key] = value
                                
                            return normalized
                    else:
                        # If we couldn't find valid JSON brackets, try a direct parse
                        try:
                            return json.loads(cleaned_result)
                        except:
                            # Last resort fallback - return a structured error response
                            return {
                                "overallAssessment": "Unable to generate a complete assessment due to formatting issues. Based on the resume information, we can provide some general guidance.",
                                "careerPathRecommendations": [
                                    {
                                        "Rank": 1,
                                        "Career Path": "General path based on skills",
                                        "Reasoning": "This recommendation is based on the skills extracted from your resume. Consider seeking personalized career advice."
                                    }
                                ],
                                "skillDevelopmentPlan": [
                                    {
                                        "Skill": "Resume Optimization",
                                        "Description": "Improve your resume format to better highlight your skills and experience",
                                        "Resources": ["Resume templates online", "Professional resume review services"]
                                    }
                                ],
                                "shortTermActions": [
                                    "Update your resume format", 
                                    "Network with professionals in your target field", 
                                    "Research companies that match your skills"
                                ],
                                "mediumTermStrategy": [
                                    "Develop core skills for your target field",
                                    "Build a professional online presence",
                                    "Seek mentorship opportunities"
                                ],
                                "longTermVision": [
                                    "Work toward leadership positions", 
                                    "Develop specialized expertise",
                                    "Consider continuous learning opportunities"
                                ]
                            }
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError in career guidance: {e}")
                    print("Cleaned response that caused error:", cleaned_result)
                    
                    # Fallback structured response when parsing fails
                    return {
                        "overallAssessment": "Unable to generate a complete assessment due to technical issues. We've provided general guidance based on your resume.",
                        "careerPathRecommendations": [
                            {
                                "Rank": 1, 
                                "Career Path": "Based on your skills", 
                                "Reasoning": "Consider roles that leverage your strongest skills identified in your resume."
                            }
                        ],
                        "skillDevelopmentPlan": [
                            {
                                "Skill": "Technical skills development",
                                "Description": "Focus on improving your technical skills based on market demand",
                                "Resources": ["Online learning platforms", "Industry certification programs"]
                            }
                        ],
                        "shortTermActions": ["Update your resume", "Network in your industry", "Identify skill gaps"],
                        "mediumTermStrategy": ["Develop missing skills", "Gain relevant experience", "Build professional relationships"],
                        "longTermVision": ["Work toward senior positions", "Develop specialization", "Consider leadership roles"]
                    }
            else:
                print("API response missing 'choices' or has empty 'choices'")
                return {
                    "error": "Failed to generate career guidance (missing response data)"
                }
                
        except Exception as e:
            print(f"Error in career guidance API call: {str(e)}")
            # Return a structured error response
            return {
                "error": str(e),
                "overallAssessment": "We encountered an error while generating your career guidance. Please try again later.",
                "careerPathRecommendations": [],
                "skillDevelopmentPlan": [],
                "shortTermActions": ["Try uploading your resume again", "Ensure your API keys are correctly configured"],
                "mediumTermStrategy": [],
                "longTermVision": []
            }
    
    
    def create_guidance_task(self, resume_data: Dict[str, Any], 
                           market_data: Dict[str, Any],
                           scoring_data: Dict[str, Any]) -> Task:
        """
        Create a CrewAI Task for career guidance.
        
        Args:
            resume_data: Resume data extracted by ResumeAnalyzerAgent
            market_data: Market trends data from JobMarketAnalyzerAgent
            scoring_data: Resume scoring data from ResumeScoringAgent
            
        Returns:
            CrewAI Task object
        """
        return Task(
            description="Provide personalized career guidance based on resume data, market trends, and resume scoring",
            expected_output="Structured career recommendations including career paths, skill development, and actionable strategies",
            agent=self.agent,
            context={
                "resume_data": resume_data,
                "market_data": market_data,
                "scoring_data": scoring_data
            }
        )
