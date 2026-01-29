# CheeseSpoon

CheeseSpoon is an intelligent course-recommender for Technion students, designed to help browse opened courses, filter eligibility, and surface course recommendations.

## Folder structure
```
.
├── cheese-spoon/
│   ├── data/                      
│   │   ├── preprocessing/         # Processed course data
│   │   └── scraping/             # Raw scraped data
│   ├── src/                      # Source code
│   │   ├── agent.py              # Main agent implementation
│   │   ├── agent_supervisor.py   # Agent coordination and routing
│   │   ├── knowledgebase.py      # Course database and search logic
│   │   └── utilities.py          # Helper functions (PDF parsing, etc.)
│   ├── static/                   # Static assets
│   │   ├── logo.png              # Application logo
│   │   └── styles.css            # Stylesheets
│   ├── templates/                # HTML templates
│   │   ├── base.html             # Base template
│   │   ├── index.html            # Landing page
│   │   ├── course_overview.html  # Course details page
│   │   ├── filters.html          # Filter configuration page
│   │   └── recommendations.html  # Results page with chat
│   ├── app.py                    # Flask application
│   ├── main.py                   # CLI entry point
│   ├── requirements.txt          # Python dependencies
│   ├── README.md                 # This file
```

## How to run

## How to run

### 1) Create and activate a virtual environment
```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate
```

### 2) Install all dependencies
```bash
pip install -r requirements.txt
```

### 3) Configure environment variables
Add the api-keys for pinecone and gemini in the `.env` file.

### 4) Start the server
```bash
flask --app app run --debug
```
or
```bash
python app.py
```
