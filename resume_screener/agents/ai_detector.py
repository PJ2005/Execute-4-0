import json
import re
import string
import math
from collections import Counter
from crewai import Agent
from typing import Dict, Any, List, Tuple
import logging
import os
import requests
from dotenv import load_dotenv

load_dotenv()

class AIDetectionAgent:
    """Agent responsible for detecting AI-generated content in resumes."""
    
    def __init__(self, openrouter_api_key=None):
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key not found. Please set OPENROUTER_API_KEY in .env file")
            
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the CrewAI agent for AI detection."""
        return Agent(
            role="AI Content Detection Specialist",
            goal="Accurately detect AI-generated content in candidate resumes",
            backstory=(
                "You are an expert in analyzing text to detect patterns consistent with AI-generated content. "
                "Your expertise allows you to identify sections of resumes that may have been written by "
                "AI tools rather than the candidate themselves."
            ),
            verbose=True,
            allow_delegation=False,
            llm_config={"model": "google/gemma-3-12b-it:free"}
        )
    
    def analyze_resume(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze resume content to detect potentially AI-generated sections.
        
        Args:
            resume_data: Processed resume data
            
        Returns:
            Dictionary with AI detection results
        """
        try:
            # Extract relevant sections to analyze
            sections = {
                "summary": resume_data.get("summary", ""),
                "experience": self._extract_text_from_experience(resume_data.get("experience", [])),
                "education": self._extract_text_from_education(resume_data.get("education", [])),
                "skills": ", ".join(resume_data.get("skills", []))
            }
            
            # Filter out empty sections
            sections = {k: v for k, v in sections.items() if v}
            
            if not sections:
                return {
                    "authenticity_score": 0.8,  # Default to relatively high authenticity
                    "flagged_sections": {},
                    "analysis": "Insufficient text to analyze for AI content detection."
                }
            
            # Analyze each section
            flagged_sections = {}
            overall_scores = []
            
            for section_name, section_text in sections.items():
                if not section_text or len(section_text) < 50:
                    # Skip sections that are too short
                    continue
                    
                result = self._detect_ai_content(section_name, section_text)
                score = result["score"]
                overall_scores.append(score)
                
                if score < 0.7:  # Flag any section with score below threshold
                    flagged_sections[section_name] = {
                        "score": score,
                        "why": result["explanation"]
                    }
            
            # Calculate overall authenticity score
            authenticity_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.8
            
            # Generate overall analysis
            analysis = self._generate_overall_analysis(flagged_sections, authenticity_score, sections)
            
            return {
                "authenticity_score": authenticity_score,
                "flagged_sections": flagged_sections,
                "analysis": analysis
            }
            
        except Exception as e:
            logging.error(f"Error detecting AI content: {e}")
            return {
                "error": f"Failed to analyze for AI-generated content: {str(e)}",
                "authenticity_score": 0.5,  # Neutral score on error
                "flagged_sections": {},
                "analysis": "Error occurred during AI content detection."
            }
    
    def _extract_text_from_experience(self, experience: List[Dict[str, Any]]) -> str:
        """Extract description text from experience entries."""
        descriptions = []
        for exp in experience[:3]:  # Only analyze the most recent 3 experiences
            if "description" in exp and exp["description"]:
                descriptions.append(exp["description"])
                
        return " ".join(descriptions)
    
    def _extract_text_from_education(self, education: List[Dict[str, Any]]) -> str:
        """Extract text from education entries."""
        return " ".join([f"{edu.get('degree', '')} in {edu.get('field', '')} from {edu.get('institution', '')}."
                      for edu in education if edu.get('degree')])
    
    def _detect_ai_content(self, section_name: str, text: str) -> Dict[str, Any]:
        """
        Detect if text is likely AI-generated.
        
        Args:
            section_name: Name of the section being analyzed
            text: Text to analyze
            
        Returns:
            Dictionary with score and explanation
        """
        prompt = f"""
        Analyze the following resume {section_name} section to determine if it was likely written by a human or generated by AI.
        Consider these factors:
        - Natural language patterns vs. formulaic expressions
        - Variety in sentence structures
        - Personal voice or unique phrasing
        - Use of specific, concrete details vs. generic statements
        - Overly perfect or formal language
        
        TEXT TO ANALYZE:
        {text[:1000]}  # Limit to 1000 chars
        
        Rate this text on a scale from 0 to 1 where:
        0 = Definitely AI-generated
        0.5 = Uncertain
        1 = Definitely human-written
        
        Format your response as JSON with these keys:
        "score": numeric value between 0-1
        "explanation": brief explanation (1-2 sentences) of your assessment
        
        Return ONLY valid JSON.
        """
        
        try:
            response_text = self._analyze_text(prompt, temperature=0.3)
            result = self._extract_json(response_text)
            
            if not result or "score" not in result:
                return {
                    "score": 0.7,  # Default to moderately authentic
                    "explanation": "Unable to determine AI content with confidence."
                }
                
            return result
            
        except Exception as e:
            logging.error(f"Error detecting AI content in {section_name}: {e}")
            return {
                "score": 0.7,  # Default to moderately authentic
                "explanation": f"Error analyzing this section: {str(e)}"
            }
    
    def _generate_overall_analysis(self, flagged_sections: Dict[str, Dict[str, Any]], 
                               authenticity_score: float, sections: Dict[str, str]) -> str:
        """
        Generate overall analysis based on flagged sections and scores.
        
        Args:
            flagged_sections: Dictionary of flagged sections
            authenticity_score: Overall authenticity score
            sections: Dictionary of resume sections
            
        Returns:
            Analysis text
        """
        prompt = f"""
        Generate a brief analysis of this resume's authenticity regarding AI-generated content.
        
        AUTHENTICITY SCORE: {authenticity_score:.2f} (0 = AI, 1 = Human)
        
        NUMBER OF FLAGGED SECTIONS: {len(flagged_sections)}
        
        FLAGGED SECTIONS:
        {json.dumps(flagged_sections, indent=2)}
        
        RESUME SECTIONS ANALYZED:
        {", ".join(sections.keys())}
        
        Provide a 3-4 sentence assessment explaining:
        1. Whether this resume appears to be AI-generated
        2. Which sections show the strongest signs of AI generation (if any)
        3. How a recruiter should interpret these results
        
        Be fair and balanced, but highlight concerning patterns if they exist.
        """
        
        try:
            analysis = self._analyze_text(prompt, temperature=0.5)
            return analysis
        except Exception as e:
            if authenticity_score < 0.4:
                return "This resume contains significant patterns consistent with AI-generated content. Multiple sections show formulaic language and generic phrasing that lack personal voice. Recruiters should exercise caution and verify skills through targeted questions."
            elif authenticity_score < 0.7:
                return "Some sections of this resume contain patterns that may indicate AI assistance. The content appears to be a mix of human and AI-generated text. Recruiters should follow up on specific claims to validate authenticity."
            else:
                return "This resume appears to be primarily human-written with natural language patterns and specific details. No significant indicators of AI-generated content were detected."
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text."""
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
            
            # Default values if JSON extraction fails
            return {
                "score": 0.7,
                "explanation": "Unable to determine AI content with confidence."
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
                        {"role": "system", "content": "You are an expert in detecting AI-generated content, providing objective analysis of text authenticity."},
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
