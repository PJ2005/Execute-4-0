import fitz  # PyMuPDF
import pdfplumber
import os
from typing import Dict, Any, Optional

class PDFParser:
    """Utility class to parse PDF resumes and extract text content."""
    
    @staticmethod
    def extract_text_with_pymupdf(file_path: str) -> str:
        """Extract text from PDF using PyMuPDF."""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print(f"Error extracting text with PyMuPDF: {str(e)}")
            return ""
            
    @staticmethod
    def extract_text_with_pdfplumber(file_path: str) -> str:
        """Extract text from PDF using pdfplumber."""
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                return text
        except Exception as e:
            print(f"Error extracting text with pdfplumber: {str(e)}")
            return ""
    
    @staticmethod
    def parse_resume(file_path: str) -> Dict[str, Any]:
        """
        Parse a resume PDF and return the extracted text.
        Tries multiple PDF parsing methods for better reliability.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        result = {
            "text": "",
            "success": False,
            "error": None
        }
        
        if not os.path.exists(file_path):
            result["error"] = "File not found"
            return result
            
        # Try primary method (PyMuPDF)
        text = PDFParser.extract_text_with_pymupdf(file_path)
        
        # If primary method fails, try backup method (pdfplumber)
        if not text:
            text = PDFParser.extract_text_with_pdfplumber(file_path)
            
        if text:
            result["text"] = text
            result["success"] = True
        else:
            result["error"] = "Failed to extract text from PDF"
            
        return result
