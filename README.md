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
pip install flask
```

### 3) Start the server
```bash
flask --app app run --debug
```
or
```bash
python app.py
```

**Note**: For now this is the mockup version, i.e. app.py is mostly statick with mock data. 
