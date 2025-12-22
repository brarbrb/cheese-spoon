from flask import Flask, render_template, request, flash, redirect, url_for
import re

app = Flask(__name__)
app.secret_key = "dev"  # change later

@app.get("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("grades_file")
        faculty = request.form.get("faculty")
        track = request.form.get("track")
        semester = request.form.get("semester")
        prefs_text = request.form.get("preferences_text")
        prefs_chips = request.form.get("preferences_chips")

        # mock behavior only (no saving)
        filename = file.filename if file and file.filename else "No file selected"
        # flash(f"Mock received: {filename} | {faculty} | {track} | {semester}")

        return redirect(url_for("eligibility"))
    return render_template("upload.html")

@app.get("/eligibility")
def eligibility():
    # mock data for now
    courses = [
        {"id": "094123", "name": "Algorithms 1", "points": 4.0, "status": "eligible", "reason": ""},
        {"id": "094210", "name": "Operating Systems", "points": 4.0, "status": "missing_prereq", "reason": "Missing: Data Structures (094134)"},
        {"id": "096345", "name": "Machine Learning", "points": 3.0, "status": "eligible", "reason": ""},
        {"id": "096400", "name": "NLP", "points": 3.0, "status": "not_offered", "reason": "Not offered this semester"},
        {"id": "094332", "name": "Databases", "points": 3.0, "status": "eligible", "reason": ""},
    ]
    summary = {"eligible": 3, "missing_prereq": 1, "not_offered": 1}
    return render_template("eligibility.html", courses=courses, summary=summary)

@app.get("/recommendations")
def recommendations():
    # mock recommendations
    recs = [
        {
            "id": "096345",
            "name": "Machine Learning",
            "points": 3.0,
            "tags": ["ML", "AI"],
            "fit": 92,
            "why": ["Matches your interest in ML/AI", "Strong review sentiment", "Good balance: points vs workload"],
            "meta": {"format": "Exam", "workload": "Medium", "lecturer": "Good at explaining"}
        },
        {
            "id": "094332",
            "name": "Databases",
            "points": 3.0,
            "tags": ["Data", "Graphs"],
            "fit": 86,
            "why": ["Useful foundation for many electives", "High academic importance (prereq)", "Reviews mention clear structure"],
            "meta": {"format": "Project", "workload": "Medium", "lecturer": "Doesn't care about attendence"}
        },
        {
            "id": "094123",
            "name": "Algorithms 1",
            "points": 4.0,
            "tags": ["Algorithms", "Core"],
            "fit": 81,
            "why": ["Bottleneck: unlocks multiple advanced courses", "Core requirement alignment", "Good teaching quality"],
            "meta": {"format": "Exam", "workload": "High", "lecturer": "Has very good lecture notes"}
        },
    ]

    prefs = {
        "chips": ["Prefer projects", "2 exams max", "Interest: ML/NLP"],
        "free_text": "I want ML/NLP courses, moderate workload, avoid 2 midterms, prefer project-based courses."
    }

    return render_template("recommendations.html", recs=recs, prefs=prefs)


def normalize_course_id(s: str) -> str:
    # keep digits only (handles spaces / dashes), keep up to 6 digits
    return re.sub(r"\D", "", s)[:6]

def build_static_course(cid: str) -> dict:
    # Static placeholder data (same shape your template expects)
    return {
        "id": cid,
        "name": "Datastructures and Algorithms",
        "points": 3.0,
        "n_reviews": 128,
        "rating_5": 4.2,
        "sentiment_100": 81,
        "tags": ["Objective", "From reviews", "Mock"],
        "summary": (
            "This is a static placeholder overview. In the real system, this text will be generated "
            "from scraped reviews + syllabus + historical metadata."
        ),
        "topics": [
            "Graphs",
            "Trees",
            "BFS/DFS",
        ],
        "pros": [
            "Well-structured Lectures",
            "''Bottlneck'' course",
            "A lot of material can be found online",
        ],
        "cons": [
            "Workload can spike near deadlines",
            "Material is hard",
            "Some reviews report uneven grading",
        ],
        "quotes": [
            "“Good course overall — requires steady weekly work.”",
            "“If you fall behind, it becomes hard quickly.”",
            "“The assignments teach a lot, but take time.”",
        ],
        "assessment": {"format": "Exam + Homework", "midterms": "None", "project": "30% of the final grade"},
        "workload": {"label": "Medium", "hours_per_week": "6–9"},
        "prereqs": ["Intro to CS"],
        "recommended_background": ["Python basics", "Descrete math"],
        "alternatives": [
            {"id": "094332", "name": "Databases"},
            {"id": "096400", "name": "Algorithms 1"},
        ],
    }

@app.route("/course-overview", methods=["GET", "POST"])
def course_overview():
    query = ""
    course = None

    if request.method == "GET":
        query = (request.args.get("course_id") or "").strip()
    else:
        query = (request.form.get("course_id") or "").strip()

    # Static rendering: if user entered anything, we always show a placeholder overview
    if query:
        cid = normalize_course_id(query) or query
        course = build_static_course(cid)

    return render_template("course_overview.html", course=course, query=query)

if __name__ == "__main__":
    app.run(debug=True)
    