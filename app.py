from flask import Flask, render_template, request, flash, redirect, url_for, session
import re
from src.utilities import parse_grades_pdf
from src.knowledgebase import recommend_courses

app = Flask(__name__)
app.secret_key = "dev"  # change later


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/course-overview")
def course_overview():
    """Display all available courses (demo view)"""
    # TODO: implement course overview logic
    # For now, you can return a simple page or redirect
    return render_template("course_overview.html")
    # OR temporarily redirect to upload:
    # return redirect(url_for('upload'))


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        # 1. Parse grade sheet
        file = request.files.get("grades_file")
        completed_course_ids = []

        if file and file.filename != '':
            completed_course_ids = parse_grades_pdf(file)
            session['completed_courses'] = completed_course_ids

        # 2. Get filter parameters
        no_exam = request.form.get("no_exam") == "true"  # checkbox value
        min_credits = float(request.form.get("min_credits", 0))

        # 3. Get ranking weights
        semantic_weight = float(request.form.get("semantic_weight", 0.2))
        credits_weight = float(request.form.get("credits_weight", 0.2))
        avg_grade_weight = float(request.form.get("avg_grade_weight", 0.2))
        workload_rating_weight = float(request.form.get("workload_rating_weight", 0.2))
        general_rating_weight = float(request.form.get("general_rating_weight", 0.2))

        # 4. Get other form data
        semester = request.form.get("semester", "WINTER_2025_2026")
        user_query = request.form.get("preferences_text", "")

        # 5. Store in session
        session['filters'] = {
            "no_exam": no_exam,
            "min_credits": min_credits,
            "semester": semester,
            "user_query": user_query
        }

        session['weights'] = {
            "semantic": semantic_weight,
            "credits": credits_weight,
            "avg_grade": avg_grade_weight,
            "workload_rating": workload_rating_weight,
            "general_rating": general_rating_weight
        }

        return redirect(url_for("recommendations"))

    return render_template("upload.html")


@app.get("/recommendations")
def recommendations():
    # Retrieve from session
    completed_courses = session.get('completed_courses', [])
    filters = session.get('filters', {})
    weights = session.get('weights', {})

    # Extract parameters
    semester = filters.get('semester', 'WINTER_2025_2026')
    no_exam = filters.get('no_exam', False)
    min_credits = filters.get('min_credits', 0)
    user_query = filters.get('user_query', '')

    # Call recommendation engine
    try:
        ranked_df = recommend_courses(
            semester_name=semester,
            courses_list=completed_courses,
            no_exam=no_exam,
            min_credits=min_credits,
            user_query=user_query,
            semantic_weight=weights.get('semantic', 0.2),
            credits_weight=weights.get('credits', 0.2),
            avg_grade_weight=weights.get('avg_grade', 0.2),
            workload_rating_weight=weights.get('workload_rating', 0.2),
            general_rating_weight=weights.get('general_rating', 0.2)
        )

        # Convert DataFrame to list of dicts for template
        courses = ranked_df.to_dict('records')

    except Exception as e:
        flash(f"Error generating recommendations: {str(e)}")
        courses = []

    # Prepare display data
    applied_filters = {
        "semester": semester,
        "no_exam": "Yes" if no_exam else "No",
        "min_credits": min_credits,
        "completed_count": len(completed_courses)
    }

    return render_template(
        "recommendations.html",
        courses=courses,
        filters=applied_filters,
        weights=weights,
        user_query=user_query
    )


def normalize_course_id(s: str) -> str:
    return re.sub(r"\D", "", s)[:6]


if __name__ == "__main__":
    app.run(debug=True)