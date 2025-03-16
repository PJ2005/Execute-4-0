# AI-Powered Resume Screening and Career Guidance

This application combines AI-powered resume screening and personalized career guidance using the CrewAI framework, OpenAI API, Serper API, and Streamlit.

## Features

- **Resume Screening**:
  - Extract key information from resumes (skills, education, experience)
  - Dynamically scrape current job postings to identify trending skills
  - Score resumes against current market demands and highlight skill gaps

- **Personalized Career Guidance**:
  - Recommend suitable career paths based on resume analysis
  - Provide actionable suggestions for skill improvement
  - Interactive conversational chatbot for personalized career queries

## Requirements

- Python 3.9+
- OpenAI API Key
- Serper API Key

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/resume-career-ai-app.git
cd resume-career-ai-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys (optional):
```
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

This will start the application on your local machine. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501).

## Deployment Instructions

### Deploying to Streamlit Cloud

1. Push your code to a GitHub repository.

2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in with your GitHub account.

3. Click "New app" and select your repository, branch, and the path to the app.py file.

4. Add your API keys as secrets in the Streamlit Cloud dashboard:
   - Go to "Advanced settings" > "Secrets"
   - Add your API keys in TOML format:
     ```toml
     OPENAI_API_KEY = "your_openai_api_key"
     SERPER_API_KEY = "your_serper_api_key"
     ```

5. Deploy the app. Streamlit Cloud will automatically install the dependencies from requirements.txt.

### Deploying to Hugging Face Spaces

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and sign in.

2. Click "Create new Space".

3. Select "Streamlit" as the SDK.

4. Link your GitHub repository or upload the files directly.

5. Add your API keys as secrets:
   - Go to "Settings" > "Repository secrets"
   - Add your API keys:
     ```
     OPENAI_API_KEY=your_openai_api_key
     SERPER_API_KEY=your_serper_api_key
     ```

6. Deploy the app. Hugging Face will automatically install the dependencies from requirements.txt.

## Application Structure

```
resume_career_ai_app/
├── agents/                 # CrewAI agents for different tasks
│   ├── __init__.py
│   ├── resume_analyzer.py  # Extracts info from resumes
│   ├── job_market_analyzer.py  # Analyzes job market trends
│   ├── resume_scorer.py    # Scores resumes against market demands
│   ├── career_guidance.py  # Provides career recommendations
│   └── chatbot_agent.py    # Interactive career advice chatbot
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── pdf_parser.py       # PDF resume parsing utilities
│   └── serper_scraper.py   # Job market scraping with Serper
├── app.py                  # Main Streamlit application
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## How It Works

1. **Resume Upload**: User uploads their resume in PDF format.

2. **Information Extraction**: The ResumeAnalyzerAgent extracts structured data from the resume.

3. **Job Market Analysis**: The JobMarketAnalyzerAgent uses Serper API to find relevant job postings and identify trending skills.

4. **Resume Scoring**: The ResumeScoringAgent compares the resume against market requirements and scores it.

5. **Career Guidance**: The CareerGuidanceAgent provides personalized recommendations based on the resume and market analysis.

6. **Interactive Chat**: The ChatbotAgent allows users to ask specific career-related questions.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
