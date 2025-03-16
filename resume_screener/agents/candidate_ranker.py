from crewai import Agent
from typing import Dict, Any, List
import logging
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

class CandidateRankingAgent:
    """Agent responsible for ranking and shortlisting candidates."""
    
    def __init__(self, openrouter_api_key=None):
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key not found. Please set OPENROUTER_API_KEY in .env file")
            
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the CrewAI agent for candidate ranking."""
        return Agent(
            role="Talent Assessment Specialist",
            goal="Rank candidates objectively based on qualifications and job fit",
            backstory=(
                "You are an expert in candidate evaluation and ranking. "
                "Your specialty is analyzing candidate profiles against job requirements "
                "to identify the best matches, ensuring fair and objective assessment."
            ),
            verbose=True,
            allow_delegation=False,
            llm_config={"model": "google/gemma-3-12b-it:free"}
        )
    
    def rank_candidates(self, 
                      job_requirements: Dict[str, Any], 
                      candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Rank candidates based on job requirements and ATS scores.
        
        Args:
            job_requirements: Job requirements dictionary
            candidates: List of candidate dictionaries with ATS scores
            
        Returns:
            Dictionary with ranked candidates and insights
        """
        try:
            if not candidates:
                return {"ranked_candidates": [], "insights": "No candidates to rank."}
                
            # Calculate composite scores and rank
            for candidate in candidates:
                # Base composite score on ATS score
                ats_score = candidate.get("ats_score", 50.0)
                
                # Add weighted components
                skill_match = candidate.get("skill_match", 50.0)
                experience_match = candidate.get("experience_match", 50.0)
                education_match = candidate.get("education_match", 50.0)
                
                # Factor in authenticity score (penalize low scores)
                authenticity_score = candidate.get("authenticity_score", 0.5)
                authenticity_factor = 0.8 + (authenticity_score * 0.4)  # Range: 0.8-1.2
                
                # Calculate composite score - higher is better
                composite_score = (
                    (ats_score * 0.5) +
                    (skill_match * 0.3) +
                    (experience_match * 0.1) +
                    (education_match * 0.1)
                ) * authenticity_factor
                
                candidate["composite_score"] = min(round(composite_score, 2), 100.0)
                
            # Sort candidates by composite score (descending)
            ranked_candidates = sorted(
                candidates, 
                key=lambda x: x.get("composite_score", 0), 
                reverse=True
            )
            
            # Generate ranking insights
            insights = self._generate_ranking_insights(job_requirements, ranked_candidates)
            
            return {
                "ranked_candidates": ranked_candidates,
                "insights": insights
            }
            
        except Exception as e:
            logging.error(f"Error ranking candidates: {e}")
            return {
                "error": str(e),
                "ranked_candidates": candidates,
                "insights": "Error generating candidate rankings."
            }
    
    def generate_comparison_data(self, 
                                job_requirements: Dict[str, Any], 
                                candidates: List[Dict[str, Any]], 
                                metrics: List[str] = None) -> Dict[str, Any]:
        """
        Generate data for comparing candidates.
        
        Args:
            job_requirements: Job requirements dictionary
            candidates: List of candidate dictionaries
            metrics: List of metrics to compare (default: use all available metrics)
            
        Returns:
            Dictionary with comparison data
        """
        if not metrics:
            metrics = ["ats_score", "skill_match", "experience_match", "education_match"]
            
        if not candidates:
            return {"comparison": {}, "key_differences": []}
            
        # Extract top candidates (max 5)
        top_candidates = candidates[:min(5, len(candidates))]
        
        # Build comparison data
        comparison = {}
        for candidate in top_candidates:
            name = candidate.get("name", "Unknown")
            comparison[name] = {}
            
            for metric in metrics:
                comparison[name][metric] = candidate.get(metric, 0)
                
            comparison[name]["matching_skills"] = candidate.get("matching_skills", [])
            comparison[name]["missing_skills"] = candidate.get("missing_skills", [])
            
        # Generate key differences
        key_differences = self._identify_key_differences(candidates, job_requirements)
        
        return {
            "comparison": comparison,
            "key_differences": key_differences,
        }
    
    def _generate_ranking_insights(self, job_requirements: Dict[str, Any], ranked_candidates: List[Dict[str, Any]]) -> str:
        """
        Generate insights about the ranked candidates.
        
        Args:
            job_requirements: Job requirements dictionary
            ranked_candidates: List of ranked candidates
            
        Returns:
            Insights text
        """
        if not ranked_candidates:
            return "No candidates available for analysis."
            
        top_candidates = ranked_candidates[:min(3, len(ranked_candidates))]
        job_title = job_requirements.get("job_title", "the position")
        
        prompt = f"""
        Analyze these candidates for {job_title}. Provide 3-4 sentences of insights about:
        
        1. How well the top candidates match the role
        2. Key strengths of the top candidates
        3. Common gaps across the candidates
        
        JOB REQUIREMENTS:
        - Required Skills: {', '.join(job_requirements.get('required_skills', []))}
        - Experience: {job_requirements.get('years_of_experience', 'Not specified')}
        - Education: {job_requirements.get('education_requirements', 'Not specified')}
        
        TOP CANDIDATES:
        {json.dumps([{
            'name': c.get('name', 'Unknown'),
            'ats_score': c.get('ats_score', 0),
            'matching_skills': c.get('matching_skills', []),
            'missing_skills': c.get('missing_skills', []),
            'experience': [{'title': e.get('title', 'Unknown'), 'duration': e.get('duration', 'Unknown')} 
                          for e in c.get('experience', [])[:2]],
            'education': [{'degree': e.get('degree', 'Unknown')} for e in c.get('education', [])[:1]]
        } for c in top_candidates], indent=2)}
        
        Keep your analysis concise and focused on meaningful observations.
        """

        try:
            insights = self._analyze_text(prompt, temperature=0.7)
            return insights
        except Exception as e:
            return f"Error generating insights: {str(e)}"
    
    def _identify_key_differences(self, candidates: List[Dict[str, Any]], job_requirements: Dict[str, Any]) -> List[str]:
        """
        Identify key differences between candidates.
        
        Args:
            candidates: List of candidate dictionaries
            job_requirements: Job requirements dictionary
            
        Returns:
            List of key differences
        """
        if not candidates or len(candidates) < 2:
            return ["Not enough candidates to compare differences."]
            
        # Extract top candidates (max 5)
        top_candidates = candidates[:min(5, len(candidates))]
        
        # Calculate average metrics
        avg_metrics = {
            "ats_score": 0,
            "skill_match": 0,
            "experience_match": 0,
            "education_match": 0,
            "experience_years": 0
        }
        
        for candidate in top_candidates:
            avg_metrics["ats_score"] += candidate.get("ats_score", 0)
            avg_metrics["skill_match"] += candidate.get("skill_match", 0)
            avg_metrics["experience_match"] += candidate.get("experience_match", 0)
            avg_metrics["education_match"] += candidate.get("education_match", 0)
            avg_metrics["experience_years"] += self._get_experience_level(candidate)
            
        for key in avg_metrics:
            avg_metrics[key] /= len(top_candidates)
            
        # Find differences
        differences = []
        
        # Compare skill sets
        all_matching_skills = set()
        all_missing_skills = set()
        
        for candidate in top_candidates:
            for skill in candidate.get("matching_skills", []):
                all_matching_skills.add(skill)
            for skill in candidate.get("missing_skills", []):
                all_missing_skills.add(skill)
        
        common_matching = set()
        for candidate in top_candidates:
            if common_matching:
                common_matching = common_matching.intersection(set(candidate.get("matching_skills", [])))
            else:
                common_matching = set(candidate.get("matching_skills", []))
                
        # Generate insights
        if common_matching:
            differences.append(f"All top candidates possess: {', '.join(list(common_matching)[:3])}")
            
        if all_missing_skills:
            differences.append(f"Skills lacking across candidates: {', '.join(list(all_missing_skills)[:3])}")
            
        # Experience difference
        max_exp = max([self._get_experience_level(c) for c in top_candidates])
        min_exp = min([self._get_experience_level(c) for c in top_candidates])
        
        if max_exp - min_exp >= 3:
            differences.append(f"Significant experience gap: from {min_exp:.1f} to {max_exp:.1f} years")
            
        return differences
    
    def _get_experience_level(self, candidate: Dict[str, Any]) -> float:
        """
        Get candidate's experience level in years.
        
        Args:
            candidate: Candidate dictionary
            
        Returns:
            Years of experience
        """
        # Try to get pre-calculated experience
        experience = candidate.get("experience", [])
        
        # Calculate total years from experience entries (simple approximation)
        total_years = 0
        for exp in experience:
            duration_str = exp.get("duration", "")
            if "year" in duration_str.lower():
                # Extract years (e.g. "2 years" -> 2)
                try:
                    import re
                    match = re.search(r'(\d+\.?\d*)\s*years?', duration_str.lower())
                    if match:
                        total_years += float(match.group(1))
                    else:
                        # Default to 1 year if we can't parse
                        total_years += 1
                except:
                    total_years += 1
                    
        return total_years
    
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
                        {"role": "system", "content": "You are an expert talent evaluation specialist who analyzes candidate profiles objectively."},
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
