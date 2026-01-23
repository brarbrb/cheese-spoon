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
    return render_template("course_overview.html")


# STEP 1: Upload grade sheet
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("grades_file")

        if not file or file.filename == '':
            flash("Please select a file to upload.")
            return redirect(url_for('upload'))

        try:
            # Parse the grade sheet
            completed_course_ids = parse_grades_pdf(file)

            if not completed_course_ids:
                flash("No courses found in the uploaded file. Please check the file and try again.")
                return redirect(url_for('upload'))

            # Store in session
            session['completed_courses'] = completed_course_ids

            # Redirect to review page
            return redirect(url_for('review_courses'))

        except Exception as e:
            flash(f"Error parsing grade sheet: {str(e)}")
            return redirect(url_for('upload'))

    return render_template("upload.html")


# STEP 2: Review and confirm courses
@app.route("/review-courses", methods=["GET", "POST"])
def review_courses():
    if request.method == "POST":
        # Get confirmed courses from form
        confirmed_courses = request.form.getlist("confirmed_courses")

        if not confirmed_courses:
            flash("Please select at least one course to continue.")
            return redirect(url_for('review_courses'))

        # Update session with confirmed courses
        session['completed_courses'] = confirmed_courses

        # Redirect to filters page
        return redirect(url_for('filters'))

    # GET request - show review page
    completed_courses = session.get('completed_courses', [])

    if not completed_courses:
        flash("No courses found. Please upload your grade sheet first.")
        return redirect(url_for('upload'))

    return render_template("review_courses.html", completed_courses=completed_courses)


# STEP 3: Set filters and weights
@app.route("/filters", methods=["GET", "POST"])
def filters():
    if request.method == "POST":
        # Get filter parameters
        no_exam = request.form.get("no_exam") == "true"
        min_credits = float(request.form.get("min_credits", 0))

        # Get ranking weights
        semantic_weight = float(request.form.get("semantic_weight", 0.2))
        credits_weight = float(request.form.get("credits_weight", 0.2))
        avg_grade_weight = float(request.form.get("avg_grade_weight", 0.2))
        workload_rating_weight = float(request.form.get("workload_rating_weight", 0.2))
        general_rating_weight = float(request.form.get("general_rating_weight", 0.2))

        # Get other form data
        semester = request.form.get("semester", "WINTER_2025_2026")
        user_query = request.form.get("preferences_text", "")

        # Store in session
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

    # GET request - show filters page
    completed_courses = session.get('completed_courses', [])

    if not completed_courses:
        flash("No courses found. Please upload your grade sheet first.")
        return redirect(url_for('upload'))

    confirmed_count = len(completed_courses)
    return render_template("filters.html", confirmed_count=confirmed_count)


# STEP 4: Get recommendations
@app.get("/recommendations")
def recommendations():
    # Retrieve from session
    completed_courses = session.get('completed_courses', [])
    filters = session.get('filters', {})
    weights = session.get('weights', {})

    if not completed_courses:
        flash("No courses found. Please upload your grade sheet first.")
        return redirect(url_for('upload'))

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