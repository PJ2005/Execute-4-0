import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from typing import List, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments like Streamlit

class VisualizationHelper:
    """Helper class for creating data visualizations for the resume screening app."""
    
    @staticmethod
    def create_ats_score_histogram(scores: List[float], bins: int = 10):
        """
        Create a histogram of ATS scores.
        
        Args:
            scores: List of ATS scores
            bins: Number of bins for the histogram
            
        Returns:
            Matplotlib figure
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(scores, bins=bins, kde=True, color='skyblue', ax=ax)
            
            ax.set_title('Distribution of ATS Scores', fontsize=16)
            ax.set_xlabel('ATS Score', fontsize=14)
            ax.set_ylabel('Number of Candidates', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add mean and median lines
            if scores:
                mean_score = np.mean(scores)
                median_score = np.median(scores)
                
                ax.axvline(mean_score, color='red', linestyle='--', 
                          label=f'Mean: {mean_score:.1f}')
                ax.axvline(median_score, color='green', linestyle='-', 
                          label=f'Median: {median_score:.1f}')
                
                ax.legend()
                
            return fig
        except Exception as e:
            import logging
            logging.error(f"Error creating ATS score histogram: {e}")
            return None
    
    @staticmethod
    def create_skill_match_radar(candidate_skills: Dict[str, float], required_skills: Dict[str, float]):
        """
        Create a radar chart comparing candidate skills with required skills.
        
        Args:
            candidate_skills: Dictionary of candidate skill -> score
            required_skills: Dictionary of required skill -> score
            
        Returns:
            Matplotlib figure
        """
        try:
            # Get union of all skills
            all_skills = list(set(candidate_skills.keys()) | set(required_skills.keys()))
            
            if not all_skills:
                return None
                
            # Create score lists (in same order)
            candidate_scores = [candidate_skills.get(skill, 0) for skill in all_skills]
            required_scores = [required_skills.get(skill, 0) for skill in all_skills]
            
            # Close the loop for the radar chart
            candidate_scores.append(candidate_scores[0])
            required_scores.append(required_scores[0])
            
            # Create angle values
            angles = np.linspace(0, 2*np.pi, len(all_skills) + 1)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            
            ax.plot(angles, candidate_scores, 'o-', linewidth=2, label='Candidate Skills', color='blue')
            ax.fill(angles, candidate_scores, alpha=0.25, color='blue')
            
            ax.plot(angles, required_scores, 'o-', linewidth=2, label='Required Skills', color='red')
            ax.fill(angles, required_scores, alpha=0.25, color='red')
            
            ax.set_thetagrids(np.degrees(angles[:-1]), all_skills)
            ax.set_title('Skill Match Analysis', fontsize=16)
            ax.grid(True)
            ax.legend(loc='upper right')
            
            plt.tight_layout()
            return fig
        except Exception as e:
            import logging
            logging.error(f"Error creating skill match radar: {e}")
            return None
    
    @staticmethod
    def create_candidate_comparison_chart(candidates: List[Dict[str, Any]], metrics: List[str] = None):
        """
        Create a bar chart comparing multiple candidates across key metrics.
        
        Args:
            candidates: List of candidate dictionaries with scores
            metrics: List of metrics to compare (defaults to standard metrics if None)
            
        Returns:
            Matplotlib figure
        """
        try:
            if not candidates or len(candidates) < 1:
                return None
                
            # Get candidate names
            names = [c.get("name", f"Candidate {i+1}") for i, c in enumerate(candidates)]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            bar_width = 0.15
            index = np.arange(len(names))
            
            # Plot each metric as a group of bars
            for i, metric in enumerate(metrics):
                values = [c.get(metric, 0) for c in candidates]
                
                # For authenticity_score, convert to percentage
                if metric == "authenticity_score":
                    values = [v * 100 for v in values]
                
                ax.bar(index + i * bar_width, values, bar_width,
                      label=metric.replace("_", " ").title())
            
            # Customize plot
            ax.set_xlabel('Candidates', fontsize=14)
            ax.set_ylabel('Score', fontsize=14)
            ax.set_title('Candidate Comparison', fontsize=16)
            ax.set_xticks(index + bar_width * (len(metrics) - 1) / 2)
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            import logging
            logging.error(f"Error creating candidate comparison chart: {e}")
            return None
    
    @staticmethod
    def create_ai_confidence_gauge(authenticity_scores: List[float]):
        """
        Create a gauge chart for AI detection confidence.
        
        Args:
            authenticity_scores: List of authenticity scores
            
        Returns:
            Matplotlib figure
        """
        try:
            if not authenticity_scores:
                return None
                
            # Convert scores to percentages for visualization
            scores_pct = [score * 100 for score in authenticity_scores]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create horizontal bars for each score
            y_pos = range(len(scores_pct))
            colors = ['red' if s < 60 else 'orange' if s < 80 else 'green' for s in scores_pct]
            
            ax.barh(y_pos, scores_pct, color=colors)
            
            # Add labels
            candidate_labels = [f"Candidate {i+1}" for i in range(len(scores_pct))]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(candidate_labels)
            
            # Add score text
            for i, score in enumerate(scores_pct):
                ax.text(min(score + 2, 95), i, f"{score:.0f}%", 
                       va='center', ha='left')
            
            # Set limits and labels
            ax.set_xlim(0, 100)
            ax.set_xlabel('Authenticity Score (%)', fontsize=14)
            ax.set_title('AI Content Detection Results', fontsize=16)
            
            # Add colored background bands
            ax.axvspan(0, 60, alpha=0.1, color='red')
            ax.axvspan(60, 80, alpha=0.1, color='orange')
            ax.axvspan(80, 100, alpha=0.1, color='green')
            
            # Add legend for the bands
            import matplotlib.patches as mpatches
            red_patch = mpatches.Patch(color='red', alpha=0.3, label='Likely AI-generated')
            orange_patch = mpatches.Patch(color='orange', alpha=0.3, label='Possibly AI-assisted')
            green_patch = mpatches.Patch(color='green', alpha=0.3, label='Likely human-written')
            ax.legend(handles=[red_patch, orange_patch, green_patch])
            
            plt.tight_layout()
            return fig
        except Exception as e:
            import logging
            logging.error(f"Error creating AI confidence gauge: {e}")
            return None
            
    @staticmethod
    def create_key_skills_chart(job_requirements: Dict[str, Any], candidates: List[Dict[str, Any]]):
        """
        Create a chart showing coverage of key skills across candidates.
        
        Args:
            job_requirements: Job requirements dictionary
            candidates: List of candidate dictionaries
            
        Returns:
            Matplotlib figure
        """
        try:
            # Extract required skills from job requirements
            required_skills = job_requirements.get("required_skills", [])
            
            if not required_skills or not candidates:
                return None
                
            # Limit to top 10 skills for readability
            required_skills = required_skills[:10]
            
            # Create data structure for plotting
            data = []
            for skill in required_skills:
                skill_data = []
                for candidate in candidates:
                    matching_skills = set(s.lower() for s in candidate.get("matching_skills", []))
                    skill_match = 1 if skill.lower() in matching_skills else 0
                    skill_data.append(skill_match)
                data.append(skill_data)
                
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create the heatmap
            skills_heatmap = ax.imshow(data, cmap='YlGn', aspect='auto')
            
            # Set labels
            ax.set_yticks(np.arange(len(required_skills)))
            ax.set_xticks(np.arange(len(candidates)))
            
            candidate_labels = [c.get("name", f"Candidate {i+1}")[:10] for i, c in enumerate(candidates)]
            ax.set_yticklabels(required_skills)
            ax.set_xticklabels(candidate_labels)
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                    rotation_mode="anchor")
            
            # Add a color bar
            cbar = plt.colorbar(skills_heatmap, ax=ax)
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['Missing', 'Present'])
            
            # Add title and labels
            ax.set_title('Required Skills Coverage', fontsize=16)
            
            # Loop over data dimensions and create text annotations
            for i in range(len(required_skills)):
                for j in range(len(candidates)):
                    text = "✓" if data[i][j] else "✗"
                    color = "white" if data[i][j] else "black"
                    ax.text(j, i, text, ha="center", va="center", color=color)
                    
            plt.tight_layout()
            return fig
        except Exception as e:
            import logging
            logging.error(f"Error creating key skills chart: {e}")
            return None
