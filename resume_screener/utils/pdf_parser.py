import fitz  # PyMuPDF
import os
import re
import logging
from typing import Dict, Any, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO)

class PDFParser:
    """Helper class for parsing PDF resumes."""
    
    @staticmethod
    def extract_resume_text(pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            text = ""
            doc = fitz.open(pdf_path)
            
            for page in doc:
                text += page.get_text()
            
            doc.close()
            return text
        except Exception as e:
            logging.error(f"Error extracting text from PDF: {e}")
            return ""
    
    @staticmethod
    def extract_contact_info(text: str) -> Dict[str, str]:
        """
        Extract contact information from resume text.
        
        Args:
            text: The resume text content
            
        Returns:
            Dictionary of contact information
        """
        contact_info = {
            "email": "",
            "phone": "",
            "location": ""
        }
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact_info["email"] = email_match.group(0)
        
        # Extract phone - improved pattern to catch more formats
        # Match various phone formats: (123) 456-7890, 123-456-7890, 123.456.7890, +1 123 456 7890
        phone_patterns = [
            r'(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}',  # Standard US formats
            r'\+\d{1,2}\s\d{3,}\s\d{3,}\s\d{3,}',                   # International format
            r'\d{3,}[-.\s]?\d{3,}[-.\s]?\d{4}'                      # Generic numeric patterns
        ]
        
        for pattern in phone_patterns:
            phone_match = re.search(pattern, text)
            if phone_match:
                contact_info["phone"] = phone_match.group(0)
                break
        
        # Enhanced location extraction with more patterns
        location_patterns = [
            r'\b[A-Z][a-zA-Z]+(?:[\s,-]+[A-Z][a-zA-Z]+)*,\s+[A-Z]{2}\b',  # City, State
            r'\b[A-Z][a-zA-Z]+(?:[\s,-]+[A-Z][a-zA-Z]+)*\b',              # Just city
            r'\b[A-Z][a-zA-Z]+(?:[\s,-]+[A-Z][a-zA-Z]+)*,\s+[A-Za-z]+\b'  # City, Country
        ]
        
        for pattern in location_patterns:
            location_match = re.search(pattern, text)
            if location_match:
                contact_info["location"] = location_match.group(0)
                break
        
        return contact_info

    @staticmethod
    def extract_skills(text: str) -> List[str]:
        """
        Extract potential skills from resume text.
        
        Args:
            text: The resume text content
            
        Returns:
            List of potential skills
        """
        skills = []
        
        # Common sections where skills might be listed
        section_patterns = [
            r"(?:SKILLS|TECHNICAL SKILLS|CORE COMPETENCIES|TECHNOLOGIES|PROFICIENCIES)[:\s]*(.*?)(?:EXPERIENCE|EDUCATION|PROJECTS|CERTIFICATIONS|\n\n|\Z)",
            r"(?:TECHNOLOGIES|TOOLS|SOFTWARE)[:\s]*(.*?)(?:EXPERIENCE|EDUCATION|PROJECTS|CERTIFICATIONS|\n\n|\Z)"
        ]
        
        for pattern in section_patterns:
            section_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if section_match:
                skills_text = section_match.group(1).strip()
                
                # Extract skills from bullet points or comma-separated lists
                if '•' in skills_text or '·' in skills_text:
                    bullet_skills = re.split(r'[•·]', skills_text)
                    for skill in bullet_skills:
                        clean_skill = skill.strip()
                        if clean_skill and len(clean_skill) > 1:
                            skills.append(clean_skill)
                elif ',' in skills_text:
                    comma_skills = skills_text.split(',')
                    for skill in comma_skills:
                        clean_skill = skill.strip()
                        if clean_skill and len(clean_skill) > 1:
                            skills.append(clean_skill)
                else:
                    # If no bullet points or commas, extract individual lines
                    line_skills = skills_text.split('\n')
                    for skill in line_skills:
                        clean_skill = skill.strip()
                        if clean_skill and len(clean_skill) > 1 and len(clean_skill) < 50:  # Avoid entire paragraphs
                            skills.append(clean_skill)
        
        # Deduplicate and clean up
        return list(set([skill for skill in skills if 2 < len(skill) < 50]))

    @staticmethod
    def extract_education(text: str) -> List[Dict[str, str]]:
        """
        Extract education information from resume.
        
        Args:
            text: The resume text content
            
        Returns:
            List of education entries
        """
        education = []
        
        # Find education section
        education_pattern = r"(?:EDUCATION|ACADEMIC BACKGROUND)[:\s]*(.+?)(?:EXPERIENCE|SKILLS|PROJECTS|CERTIFICATIONS|\Z)"
        education_match = re.search(education_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if education_match:
            education_text = education_match.group(1).strip()
            
            # Extract degree information
            degree_patterns = [
                r"((?:Bachelor|Master|Ph\.?D|Doctorate|B\.S\.|M\.S\.|B\.A\.|M\.A\.|MBA|Associates)[\s\w\.]+)(?:\sin\s|\s-\s|\s|\,)([^,\n]*)(?:[\s,]*|[\s,]at[\s,]*)([^,\n]*)",
                r"([^,\n]*)(?:University|College|Institute|School)([^,\n]*)"
            ]
            
            for pattern in degree_patterns:
                for match in re.finditer(pattern, education_text, re.IGNORECASE):
                    if len(match.groups()) >= 3:
                        degree = match.group(1).strip()
                        field = match.group(2).strip()
                        institution = match.group(3).strip()
                        
                        # Extract graduation date
                        date_pattern = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[a-z]*[\s,-]*\d{4}"
                        date_match = re.search(date_pattern, education_text, re.IGNORECASE)
                        graduation_date = date_match.group(0).strip() if date_match else ""
                        
                        education.append({
                            "degree": degree,
                            "field": field,
                            "institution": institution,
                            "graduation_date": graduation_date
                        })
                    elif len(match.groups()) >= 2:
                        institution = match.group(1).strip() + " " + match.group(2).strip()
                        
                        # Try to extract degree information from nearby text
                        degree_text_pattern = r"(?:Bachelor|Master|Ph\.?D|Doctorate|B\.S\.|M\.S\.|B\.A\.|M\.A\.|MBA|Associates)[\s\w\.]+"
                        degree_text_match = re.search(degree_text_pattern, education_text, re.IGNORECASE)
                        degree = degree_text_match.group(0).strip() if degree_text_match else "Degree"
                        
                        education.append({
                            "degree": degree,
                            "institution": institution,
                            "graduation_date": ""
                        })
        
        return education

    @staticmethod
    def extract_experience(text: str) -> List[Dict[str, str]]:
        """
        Extract work experience from resume.
        
        Args:
            text: The resume text content
            
        Returns:
            List of work experience entries
        """
        experience = []
        
        # Find experience section
        experience_pattern = r"(?:EXPERIENCE|WORK EXPERIENCE|EMPLOYMENT|PROFESSIONAL EXPERIENCE)[:\s]*(.+?)(?:EDUCATION|SKILLS|PROJECTS|CERTIFICATIONS|\Z)"
        experience_match = re.search(experience_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if experience_match:
            experience_text = experience_match.group(1).strip()
            
            # Split by potential job entries (company names, dates)
            # This is a simplified approach - real-world resumes may require more complex parsing
            job_entries = re.split(r'\n\s*\n', experience_text)
            
            for entry in job_entries:
                if not entry.strip():
                    continue
                    
                # Extract job title and company
                title_pattern = r"(?:^|\n)([^,\n\|]+)(?:[\s,]*|[\s,]at[\s,]*|[\s,]*\|[\s,]*)([^,\n]*)"
                title_match = re.search(title_pattern, entry)
                
                if title_match:
                    title = title_match.group(1).strip()
                    company = title_match.group(2).strip()
                    
                    # Extract dates
                    date_pattern = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[a-z]*[\s,-]*\d{4}(?:\s*(?:-|to|–|—)\s*(?:Present|Current|Now|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[a-z]*[\s,]*\d{0,4})?"
                    date_match = re.search(date_pattern, entry, re.IGNORECASE)
                    dates = date_match.group(0).strip() if date_match else ""
                    
                    # Split into start and end dates if possible
                    if dates:
                        date_parts = re.split(r'\s*(?:-|to|–|—)\s*', dates)
                        start_date = date_parts[0].strip() if date_parts else ""
                        end_date = date_parts[1].strip() if len(date_parts) > 1 else "Present"
                    else:
                        start_date = ""
                        end_date = ""
                    
                    # Calculate duration (rough estimate)
                    duration = "Not specified"
                    if start_date and end_date:
                        # Extract years
                        start_year_match = re.search(r'\d{4}', start_date)
                        end_year_match = re.search(r'\d{4}', end_date) if end_date != "Present" else None
                        
                        if start_year_match:
                            start_year = int(start_year_match.group(0))
                            end_year = int(end_year_match.group(0)) if end_year_match else 2023  # Default current year
                            years = end_year - start_year
                            duration = f"{years} year{'s' if years != 1 else ''}"
                    
                    # Extract description (everything after the job title/company/dates)
                    lines = entry.split('\n')
                    description_lines = []
                    capture = False
                    
                    for line in lines:
                        if not capture:
                            if dates in line:
                                capture = True
                                continue
                        else:
                            description_lines.append(line.strip())
                    
                    description = "\n".join(description_lines)
                    
                    experience.append({
                        "title": title,
                        "company": company,
                        "start_date": start_date,
                        "end_date": end_date,
                        "duration": duration,
                        "description": description
                    })
        
        return experience

    @staticmethod
    def extract_certifications(text: str) -> List[str]:
        """
        Extract certifications from resume.
        
        Args:
            text: The resume text content
            
        Returns:
            List of certifications
        """
        certifications = []
        
        # Find certifications section
        cert_pattern = r"(?:CERTIFICATIONS|CERTIFICATES|LICENSES)[:\s]*(.+?)(?:EXPERIENCE|EDUCATION|SKILLS|PROJECTS|\Z)"
        cert_match = re.search(cert_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if cert_match:
            cert_text = cert_match.group(1).strip()
            
            # Extract individual certifications
            # Assume they're separated by new lines or bullet points
            if '•' in cert_text or '·' in cert_text:
                bullet_certs = re.split(r'[•·]', cert_text)
                for cert in bullet_certs:
                    clean_cert = cert.strip()
                    if clean_cert and len(clean_cert) > 3:
                        certifications.append(clean_cert)
            else:
                # If no bullet points, extract individual lines
                line_certs = cert_text.split('\n')
                for cert in line_certs:
                    clean_cert = cert.strip()
                    if clean_cert and len(clean_cert) > 3 and len(clean_cert) < 100:  # Avoid entire paragraphs
                        certifications.append(clean_cert)
        
        return certifications
    
    @staticmethod
    def parse_resume(file_path: str) -> Dict[str, Any]:
        """
        Parse a resume file and extract key information.
        
        Args:
            file_path: Path to the resume file
            
        Returns:
            Dictionary with extracted information
        """
        try:
            if not os.path.exists(file_path):
                return {"success": False, "error": f"File not found: {file_path}"}
                
            _, ext = os.path.splitext(file_path)
            if ext.lower() != '.pdf':
                return {"success": False, "error": "Only PDF files are supported"}
                
            text = PDFParser.extract_resume_text(file_path)
            
            if not text:
                return {"success": False, "error": "Could not extract text from PDF"}
                
            # Extract structured information
            contact_info = PDFParser.extract_contact_info(text)
            skills = PDFParser.extract_skills(text)
            education = PDFParser.extract_education(text)
            experience = PDFParser.extract_experience(text)
            certifications = PDFParser.extract_certifications(text)
            
            # Try to extract name (very basic)
            name = "Unknown"
            name_match = re.search(r'^([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2})', text)
            if name_match:
                name = name_match.group(1)
                
            # Prepare full structure similar to resume_career_ai_app
            structured_data = {
                "name": name,
                "contact_info": contact_info,
                "skills": skills,
                "education": education,
                "experience": experience,
                "certifications": certifications,
                "summary": "",  # Hard to reliably extract
                "languages": [],  # Would need more sophisticated extraction
            }
            
            return {
                "success": True,
                "text": text,
                "contact_info": contact_info,
                "structured_data": structured_data
            }
        except Exception as e:
            logging.error(f"Error parsing resume: {e}")
            return {"success": False, "error": str(e)}
