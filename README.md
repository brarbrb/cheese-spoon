# CheeseSpoon

CheeseSpoon is an intelligent course-recommender for Technion students, designed to help browse opened courses, filter eligibility, and surface course recommendations.

## Folder structure
```
.
├── app.py
├── templates/
│   ├── index.html
│   ├── upload.html
│   ├── eligibility.html
│   ├── recommendations.html
│   └── course_overview.html
└── static/
    ├── styles.css
    └── logo.png
```

## How to run

### 1) Create and activate a virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Start the server
```bash
flask --app app run --debug
```

Open: http://127.0.0.1:5000
