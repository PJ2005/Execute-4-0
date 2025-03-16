# AI-Powered Resume Screening Tool

An advanced resume screening application for recruiters that uses AI to analyze job descriptions, score candidate resumes against requirements, detect AI-generated content, and provide insightful candidate rankings.

## Features

- **Job Description Analysis**: Extract key skills, requirements, and qualifications from job postings
- **Resume Batch Processing**: Process multiple candidate resumes simultaneously
- **ATS Scoring & Matching**: Score resumes against job requirements with industry-standard ATS algorithms
- **AI Content Detection**: Flag potentially AI-generated resume content
- **Candidate Shortlisting**: Generate ranked shortlists of top candidates
- **Job Market Insights**: Get real-time job market data relevant to your search

## 🔧 Technology Stack

- Python 3.9+
- OpenRouter API with Gemma3-12b-it model
- Streamlit for the user interface
- Serper API for job market insights
- CrewAI for agent-based processing
- PyMuPDF for document processing

## 📋 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/resume-screener.git
   cd resume-screener
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with the following:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key
   SERPER_API_KEY=your_serper_api_key  # Optional, for job market insights
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## 🚀 Usage

1. **Upload Job Description**: Paste a job description or upload a file
2. **Upload Resumes**: Upload one or more candidate resumes (PDF format)
3. **Review Analysis**: Examine job requirements, candidate rankings, and detailed insights
4. **Export Results**: Download candidate rankings and analysis (coming soon)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

