import json
import re
import logging
from typing import Dict, Any, Union, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)

class JsonHandler:
    """Utility class for safely handling JSON operations."""
    
    @staticmethod
    def safe_get(data: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """
        Safely get a value from nested dictionary using dot notation.
        
        Args:
            data: Dictionary to search in
            key_path: Path to the value using dot notation (e.g., 'person.address.city')
            default: Default value to return if path doesn't exist
            
        Returns:
            Value at the key path or the default value
        """
        if not data or not isinstance(data, dict):
            return default
            
        keys = key_path.split('.')
        current = data
        
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
            
        return current if current is not None else default
    
    @staticmethod
    def extract_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from text that might contain other content.
        
        Args:
            text: Text that might contain JSON
            
        Returns:
            Extracted JSON as dictionary or None if extraction fails
        """
        try:
            # First try: direct JSON parsing
            return json.loads(text)
        except json.JSONDecodeError:
            # Second try: find JSON object in text
            try:
                json_match = re.search(r'(\{.*\})', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
                
            # Third try: find JSON in code blocks
            try:
                if "```json" in text:
                    json_block = text.split("```json")[1].split("```")[0].strip()
                    return json.loads(json_block)
                elif "```" in text:
                    json_block = text.split("```")[1].strip()
                    return json.loads(json_block)
            except (json.JSONDecodeError, IndexError):
                pass
                
        return None
    
    @staticmethod
    def merge_json(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two JSON objects, with values from 'updates' taking precedence.
        
        Args:
            base: Base dictionary
            updates: Dictionary with update values
            
        Returns:
            Merged dictionary
        """
        if not isinstance(base, dict) or not isinstance(updates, dict):
            return updates if updates is not None else base
            
        result = base.copy()
        
        for key, value in updates.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                # Recursive merge for nested dictionaries
                result[key] = JsonHandler.merge_json(result[key], value)
            else:
                # Direct update for non-dict values or keys not in base
                result[key] = value
                
        return result
    
    @staticmethod
    def clean_none_values(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove None values from dictionary to prevent TypeError when serializing.
        
        Args:
            data: Dictionary that might contain None values
            
        Returns:
            Dictionary with None values removed
        """
        if not isinstance(data, dict):
            return data
            
        result = {}
        for key, value in data.items():
            if value is None:
                continue
            elif isinstance(value, dict):
                result[key] = JsonHandler.clean_none_values(value)
            elif isinstance(value, list):
                result[key] = [
                    JsonHandler.clean_none_values(item) if isinstance(item, dict) else item
                    for item in value if item is not None
                ]
            else:
                result[key] = value
                
        return result
    
    @staticmethod
    def ensure_valid_json(data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        Ensure data is valid JSON by parsing if it's a string or cleaning if it's a dict.
        
        Args:
            data: String JSON or dictionary
            
        Returns:
            Clean dictionary with no None values
        """
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                extracted = JsonHandler.extract_json(data)
                if extracted:
                    data = extracted
                else:
                    return {}  # Return empty dict if extraction fails
                
        if not isinstance(data, dict):
            return {}
            
        return JsonHandler.clean_none_values(data)
