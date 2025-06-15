# HR Recruiter - Resume Analysis Tool

An intelligent resume analysis tool that uses AI to evaluate candidate resumes against job descriptions. The tool integrates with Gmail to fetch resumes and provides detailed analysis using OpenAI's GPT model.

## Features

- Gmail integration for fetching resumes
- AI-powered resume analysis
- Detailed matching scores and insights
- Support for multiple file formats (PDF, DOCX)
- Progress tracking and logging
- CSV export of analysis results

## Prerequisites

- Python 3.8 or higher
- Gmail account with API access
- OpenAI API key

## Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/hr-recruiter.git
cd hr-recruiter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Gmail API:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project
   - Enable Gmail API
   - Create OAuth 2.0 credentials
   - Download the credentials and save as `credentials.json`

4. Set up OpenAI API:
   - Get your API key from [OpenAI](https://platform.openai.com/)
   - Set it as an environment variable:
     ```bash
     # Windows
     set OPENAI_API_KEY=your_api_key_here
     
     # Linux/Mac
     export OPENAI_API_KEY=your_api_key_here
     ```

## Usage

1. Run the script:
```bash
python gmail_processor.py
```

2. Enter the Gmail label name when prompted
3. Paste the job description when prompted
4. View the analysis results in the console and CSV file

## Configuration

- Modify `config.py` to adjust:
  - File paths
  - Logging settings
  - Analysis parameters

## Security

- Never commit sensitive information like API keys
- Use environment variables for credentials
- Keep your `credentials.json` and `token.json` files secure

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 