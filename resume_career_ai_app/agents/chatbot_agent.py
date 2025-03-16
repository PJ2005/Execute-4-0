from crewai import Agent, Task
import openai
from typing import Dict, Any, List
import json
from dotenv import load_dotenv
import os
import requests

class ChatbotAgent:
    """Agent responsible for interactive career-related conversations"""
    
    def __init__(self, openrouter_api_key: str):
        """
        Initialize the ChatbotAgent.
        
        Args:
            openrouter_api_key: OpenRouter API key for model access
        """
        self.openrouter_api_key = openrouter_api_key
        
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key not provided")
        
        # Create the CrewAI agent
        self.agent = Agent(
            role="Career Advisor Chatbot",
            goal="Provide helpful career advice through conversational interaction",
            backstory=(
                "As an AI career advisor, you excel at understanding career-related questions "
                "and providing helpful, personalized advice. You have deep knowledge of various "
                "industries, job roles, skills, and career development strategies. Your goal is "
                "to assist users with their career questions in a supportive and informative manner."
            ),
            verbose=True,
            allow_delegation=False,
            llm_config={"model": "google/gemma-3-12b-it:free"}
        )
        
        # Initialize conversation history
        self.conversation_history = []
    
    def get_response(self, user_message: str, 
                   resume_data: Dict[str, Any] = None,
                   career_guidance: Dict[str, Any] = None) -> str:
        """
        Get a response from the chatbot based on user message and context.
        
        Args:
            user_message: The user's message
            resume_data: Optional resume data for context
            career_guidance: Optional career guidance data for context
            
        Returns:
            Chatbot response as a string
        """
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Prepare context information
        context = ""
        if resume_data:
            context += f"Resume Information: {json.dumps(resume_data)}\n\n"
        if career_guidance:
            context += f"Career Guidance: {json.dumps(career_guidance)}\n\n"
        
        system_prompt = f"""
        You are a helpful career advisor chatbot. Provide concise, personalized career advice based on the user's questions.
        
        Context information (use this to personalize your answers if relevant):
        {context}
        
        Guidelines:
        - Keep responses concise and direct (3-4 sentences at most)
        - Be supportive and encouraging
        - Provide specific, actionable advice when possible
        - If you don't know something, admit it rather than making up information
        - Focus only on career-related questions
        """
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (limited to last 10 exchanges to manage context)
        history_limit = min(10, len(self.conversation_history))
        messages.extend(self.conversation_history[-history_limit:])
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "google/gemma-3-4b-it:free",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 250
                }
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            chatbot_response = response_data["choices"][0]["message"]["content"]
            
            # Add bot response to conversation history
            self.conversation_history.append({"role": "assistant", "content": chatbot_response})
            
            return chatbot_response
                
        except Exception as e:
            print(f"Error in OpenRouter API call: {str(e)}")
            return "I'm sorry, I encountered an error. Could you try asking again in a different way?"
    
    def create_chatbot_task(self, user_message: str, context: Dict[str, Any]) -> Task:
        """
        Create a CrewAI Task for chatbot interaction.
        
        Args:
            user_message: The user's message
            context: Dictionary with context information
            
        Returns:
            CrewAI Task object
        """
        return Task(
            description=f"Respond to the user's message: '{user_message}'",
            expected_output="Conversational response providing helpful career advice",
            agent=self.agent,
            context=context
        )
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
