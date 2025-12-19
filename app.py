from flask import Flask, render_template, request, flash, redirect, url_for

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
        flash(f"Mock received: {filename} | {faculty} | {track} | {semester}")

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
            "meta": {"format": "Exam", "workload": "Medium", "lecturer": "High"}
        },
        {
            "id": "094332",
            "name": "Databases",
            "points": 3.0,
            "tags": ["Data", "Systems"],
            "fit": 86,
            "why": ["Useful foundation for many electives", "High academic importance (prereq)", "Reviews mention clear structure"],
            "meta": {"format": "Project", "workload": "Medium", "lecturer": "Medium–High"}
        },
        {
            "id": "094123",
            "name": "Algorithms 1",
            "points": 4.0,
            "tags": ["Algorithms", "Core"],
            "fit": 81,
            "why": ["Bottleneck: unlocks multiple advanced courses", "Core requirement alignment", "Good teaching quality"],
            "meta": {"format": "Exam", "workload": "High", "lecturer": "Medium"}
        },
    ]

    prefs = {
        "chips": ["Prefer projects", "2 exams max", "Interest: ML/NLP"],
        "free_text": "I want ML/NLP courses, moderate workload, avoid 2 midterms, prefer project-based courses."
    }

    return render_template("recommendations.html", recs=recs, prefs=prefs)

if __name__ == "__main__":
    app.run(debug=True)
    