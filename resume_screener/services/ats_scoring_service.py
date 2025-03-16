import logging
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dotenv import load_dotenv

from agents.job_analyzer import JobAnalyzerAgent
from agents.resume_processor import ResumeProcessorAgent
from agents.ats_scorer import ATSScoringAgent
from agents.ai_detector import AIDetectionAgent
from agents.candidate_ranker import CandidateRankingAgent

load_dotenv()

class ATSScoringService:
    """Service for scoring resumes against job descriptions."""
    
    def __init__(self):
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key not found. Please set OPENROUTER_API_KEY in .env file")
            
        self.job_analyzer = JobAnalyzerAgent(self.openrouter_api_key)
        self.resume_processor = ResumeProcessorAgent(self.openrouter_api_key)
        self.ats_scorer = ATSScoringAgent(self.openrouter_api_key)
        self.ai_detector = AIDetectionAgent(self.openrouter_api_key)
        self.candidate_ranker = CandidateRankingAgent(self.openrouter_api_key)
    
    def analyze_job_description(self, job_description: str) -> Dict[str, Any]:
        """
        Analyze a job description and extract requirements.
        
        Args:
            job_description: The job description text
            
        Returns:
            Dictionary with structured job requirements
        """
        return self.job_analyzer.analyze_job_description(job_description)
    
    def process_resume(self, resume_text: str) -> Dict[str, Any]:
        """
        Process a resume and extract structured information.
        
        Args:
            resume_text: The resume text
            
        Returns:
            Dictionary with structured resume data
        """
        return self.resume_processor.process_resume(resume_text)
    
    def batch_process_resumes(self, resume_files: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple resumes in parallel.
        
        Args:
            resume_files: Dictionary mapping file names to resume text
            
        Returns:
            Dictionary mapping file names to processed resume data
        """
        results = {}
        
        with ThreadPoolExecutor() as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_resume, resume_text): file_name
                for file_name, resume_text in resume_files.items()
            }
            
            # Process results as they complete
            for future in as_completed(future_to_file):
                file_name = future_to_file[future]
                try:
                    results[file_name] = future.result()
                except Exception as e:
                    logging.error(f"Error processing resume {file_name}: {e}")
                    results[file_name] = {"error": str(e)}
                    
        return results
    
    def score_resume(self, job_requirements: Dict[str, Any], resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a resume against job requirements.
        
        Args:
            job_requirements: The job requirements dictionary
            resume_data: The processed resume data
            
        Returns:
            Dictionary with ATS scores
        """
        return self.ats_scorer.score_resume(job_requirements, resume_data)
    
    def batch_score_resumes(self, job_requirements: Dict[str, Any], 
                          processed_resumes: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Score multiple resumes against job requirements in parallel.
        
        Args:
            job_requirements: The job requirements dictionary
            processed_resumes: Dictionary mapping file names to processed resume data
            
        Returns:
            Dictionary mapping file names to ATS scores
        """
        results = {}
        
        with ThreadPoolExecutor() as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.score_resume, job_requirements, resume_data): file_name
                for file_name, resume_data in processed_resumes.items()
                if "error" not in resume_data
            }
            
            # Process results as they complete
            for future in as_completed(future_to_file):
                file_name = future_to_file[future]
                try:
                    results[file_name] = future.result()
                except Exception as e:
                    logging.error(f"Error scoring resume {file_name}: {e}")
                    results[file_name] = {"error": str(e)}
                    
        return results
    
    def detect_ai_content(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect potentially AI-generated content in a resume.
        
        Args:
            resume_data: The processed resume data
            
        Returns:
            Dictionary with AI detection results
        """
        try:
            return self.ai_detector.analyze_resume(resume_data)
        except AttributeError:
            # Fallback if AI detector isn't properly implemented
            return {
                "authenticity_score": 0.8,  # Default to high authenticity
                "flagged_sections": {},
                "analysis": "AI content detection not available."
            }
    
    def batch_detect_ai_content(self, processed_resumes: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Detect AI-generated content in multiple resumes in parallel.
        
        Args:
            processed_resumes: Dictionary mapping file names to processed resume data
            
        Returns:
            Dictionary mapping file names to AI detection results
        """
        results = {}
        
        with ThreadPoolExecutor() as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.detect_ai_content, resume_data): file_name
                for file_name, resume_data in processed_resumes.items()
                if "error" not in resume_data
            }
            
            # Process results as they complete
            for future in as_completed(future_to_file):
                file_name = future_to_file[future]
                try:
                    results[file_name] = future.result()
                except Exception as e:
                    logging.error(f"Error detecting AI content in {file_name}: {e}")
                    results[file_name] = {"error": str(e), "authenticity_score": 0.5}
                    
        return results
    
    def process_job_and_resumes(self, job_description: str, resume_files: Dict[str, str]) -> Dict[str, Any]:
        """
        Complete end-to-end processing of job description and resumes.
        
        Args:
            job_description: The job description text
            resume_files: Dictionary mapping file names to resume text
            
        Returns:
            Dictionary with complete analysis results
        """
        try:
            # Step 1: Analyze job description
            job_requirements = self.analyze_job_description(job_description)
            
            # Step 2: Process resumes
            processed_resumes = self.batch_process_resumes(resume_files)
            
            # Step 3: Score resumes against job requirements
            ats_scores = self.batch_score_resumes(job_requirements, processed_resumes)
            
            # Step 4: Detect AI content
            ai_detection_results = self.batch_detect_ai_content(processed_resumes)
            
            # Step 5: Prepare candidate data for ranking
            candidates = []
            for file_name, resume_data in processed_resumes.items():
                candidate = {
                    "file_name": file_name,
                    "name": resume_data.get("name", "Unknown"),
                    **resume_data,
                }
                
                # Add ATS scores
                if file_name in ats_scores:
                    candidate.update(ats_scores[file_name])
                    
                # Add AI detection results
                if file_name in ai_detection_results:
                    candidate.update({
                        "authenticity_score": ai_detection_results[file_name].get("authenticity_score", 0.5),
                        "flagged_sections": ai_detection_results[file_name].get("flagged_sections", {}),
                        "ai_analysis": ai_detection_results[file_name].get("analysis", "")
                    })
                    
                candidates.append(candidate)
                
            # Step 6: Rank candidates
            ranking_results = self.candidate_ranker.rank_candidates(job_requirements, candidates)
            
            # Step 7: Generate comparison data
            comparison_data = self.candidate_ranker.generate_comparison_data(
                job_requirements,
                ranking_results.get("ranked_candidates", [])
            )
            
            # Add comparison data to ranking results
            ranking_results.update(comparison_data)
            
            return {
                "job_requirements": job_requirements,
                "candidates": candidates,
                "ranking_results": ranking_results
            }
            
        except Exception as e:
            logging.error(f"Error in end-to-end processing: {e}")
            return {
                "error": str(e),
                "job_requirements": {},
                "candidates": [],
                "ranking_results": {}
            }
