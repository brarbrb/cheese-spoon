from flask import Flask, render_template, request, flash, redirect, url_for, session, jsonify
import re, json
from src.utilities import parse_grades_pdf, parse_review_summary, normalize_course_id, clean_description
from src.knowledgebase import recommend_courses, get_course_by_id
from src.agent_supervisor import supervisor_agent  # Use new supervisor

app = Flask(__name__)
app.secret_key = "dev"  # change later


@app.get("/")
def index():
    return render_template("index.html")


@app.route("/course-overview", methods=["GET", "POST"])
def course_overview():
    """Display specific course details and structured summary"""
    course_data = None
    query = ""
    alternatives = []

    if request.method == "POST":
        # Get query and clean it
        query = request.form.get("course_id", "").strip()
        clean_id = re.sub(r"\D", "", query)

        if clean_id:
            try:
                # 1. Fetch metadata from Pinecone
                semester = session.get('filters', {}).get('semester', 'WINTER_2025_2026')
                raw_data = get_course_by_id(clean_id, semester)

                if raw_data:
                    # 2. Parse the review summary string into structured parts
                    summary_parts = parse_review_summary(raw_data.get('reviews_summary', ''))

                    # 3. Process the summaries into (Overview, Quotes)
                    interest_data = split_summary_and_quotes(summary_parts.get('interest', ''))
                    workload_data = split_summary_and_quotes(summary_parts.get('workload', ''))
                    bottom_data = split_summary_and_quotes(summary_parts.get('bottom_line', ''))

                    # 4. Calculate average grade from the JSON dictionary
                    avg_grade = 0
                    try:
                        grades_dict = json.loads(raw_data.get('avg_grades', '{}'))
                        if grades_dict:
                            avg_grade = sum(grades_dict.values()) / len(grades_dict)
                    except:
                        avg_grade = 0

                    # 5. Parse prerequisites
                    prereqs = []
                    try:
                        prereqs = json.loads(raw_data.get('prerequisites', '[]'))
                        # Flatten list of lists for display if simple, or keep structure
                        flat_prereqs = []
                        for group in prereqs:
                            flat_prereqs.append(" OR ".join(group))
                        prereqs = flat_prereqs
                    except:
                        prereqs = []

                    # 6. Build the display object
                    course_data = {
                        "id": raw_data.get('id', clean_id),
                        "name": raw_data.get('title', 'Unknown Course'),
                        "points": float(raw_data.get('credits', 0)),
                        "rating_5": float(raw_data.get('general_rating', 0) or 0),
                        "workload_rating": float(raw_data.get('workload_rating', 0) or 0),

                        # Structured Summaries (Now Dicts with 'overview' and 'quotes')
                        "summary_interest": interest_data,
                        "summary_workload": workload_data,
                        "summary_bottom_line": bottom_data,
                        "description": clean_description(raw_data.get('description', '')),

                        # Facts
                        "prereqs": prereqs,
                        "moed_a": raw_data.get('moed_a'),
                        "moed_b": raw_data.get('moed_b'),
                        "avg_grade": avg_grade
                    }

                    # 7. Get "Similar Courses" (Alternatives)
                    try:
                        recs_df = recommend_courses(
                            semester_name=semester,
                            courses_list=[clean_id],  # Exclude current course
                            user_query=course_data['name'],  # Search by title
                            semantic_weight=0.9,  # High weight on semantic match
                            credits_weight=0,
                            avg_grade_weight=0.1,
                            workload_rating_weight=0,
                            general_rating_weight=0
                        )

                        if not recs_df.empty:
                            alternatives = recs_df.head(3).to_dict('records')
                    except Exception as e:
                        print(f"Error fetching alternatives: {e}")
                else:
                    flash(f"Course {clean_id} not found in database.")

            except Exception as e:
                print(f"Overview Error: {e}")
                flash("An error occurred while fetching course details.")

    return render_template("course_overview.html", course=course_data, query=query, alternatives=alternatives)


# STEP 1: Upload grade sheet
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("grades_file")

        if not file or file.filename == '':
            session['completed_courses'] = []
            flash("Skipped upload. You can manually add courses now.")
            return redirect(url_for('review_courses'))

        try:
            # Returns: [{'id': '00940210', 'name': 'Computer Org'}, ...]
            parsed_courses = parse_grades_pdf(file)

            if not parsed_courses:
                flash("No courses found. Try a different file.")
                return redirect(url_for('upload'))

            session['completed_courses'] = parsed_courses
            return redirect(url_for('review_courses'))

        except Exception as e:
            print(f"Upload Error: {e}")
            flash(f"Error parsing file: {str(e)}")
            return redirect(url_for('upload'))
    return render_template("upload.html")


# STEP 2: Review and confirm courses
@app.route("/review_courses", methods=["GET", "POST"])
def review_courses():
    if 'completed_courses' not in session:
        session['completed_courses'] = []

    current_courses = session.get('completed_courses', [])

    if request.method == "POST":
        # --- CASE 1: MANUAL ADD ---
        if "new_course_id" in request.form:
            new_id = request.form.get("new_course_id").strip()
            # Get the name (default to the ID if left empty)
            new_name = request.form.get("new_course_name", "").strip() or new_id

            existing_ids = {c['id'] for c in current_courses}

            if new_id and new_id not in existing_ids:
                # Save both ID and Name
                current_courses.append({'id': new_id, 'name': new_name})
                session['completed_courses'] = current_courses
                flash(f"Added course {new_id}: {new_name}")
            elif new_id in existing_ids:
                flash(f"Course {new_id} is already in the list.")

            return redirect(url_for('review_courses'))

        # --- CASE 2: CONFIRM SELECTION ---
        confirmed_ids = request.form.getlist("confirmed_courses")
        filtered_courses = [c for c in current_courses if c['id'] in confirmed_ids]
        session['completed_courses'] = filtered_courses

        return redirect(url_for('filters'))

    return render_template("review_courses.html", completed_courses=current_courses)


# STEP 3: Set filters and weights
@app.route("/filters", methods=["GET", "POST"])
def filters():
    if request.method == "POST":
        # Get filter parameters
        no_exam = request.form.get("no_exam") == "true"
        min_credits = float(request.form.get("min_credits", 0))

        # Get importance values (1-5 scale)
        semantic_importance = float(request.form.get("semantic_importance", 3))
        credits_importance = float(request.form.get("credits_importance", 3))
        avg_grade_importance = float(request.form.get("avg_grade_importance", 3))
        workload_rating_importance = float(request.form.get("workload_rating_importance", 3))
        general_rating_importance = float(request.form.get("general_rating_importance", 3))

        # Normalize importance values to weights that sum to 1.0
        total_importance = (semantic_importance + credits_importance +
                            avg_grade_importance + workload_rating_importance +
                            general_rating_importance)

        if total_importance == 0:
            total_importance = 1  # Avoid division by zero

        weights = {
            'semantic': round(semantic_importance / total_importance, 3),
            'credits': round(credits_importance / total_importance, 3),
            'avg_grade': round(avg_grade_importance / total_importance, 3),
            'workload_rating': round(workload_rating_importance / total_importance, 3),
            'general_rating': round(general_rating_importance / total_importance, 3)
        }

        # Get user query
        user_query = request.form.get("preferences_text", "").strip()

        # Get semester
        semester = request.form.get("semester", "WINTER_2025_2026")

        # Store everything in session
        session['filters'] = {
            'semester': semester,
            'no_exam': no_exam,
            'min_credits': min_credits
        }
        session['weights'] = weights
        session['user_query'] = user_query

        # Store importance values for UI
        session['importance'] = {
            'semantic': int(semantic_importance),
            'credits': int(credits_importance),
            'avg_grade': int(avg_grade_importance),
            'workload_rating': int(workload_rating_importance),
            'general_rating': int(general_rating_importance)
        }

        return redirect(url_for('recommendations'))

    # GET: Show filter page with defaults
    if 'filters' not in session:
        session['filters'] = {
            'semester': 'WINTER_2025_2026',
            'no_exam': False,
            'min_credits': 0
        }
    if 'weights' not in session:
        session['weights'] = {
            'semantic': 0.2, 'credits': 0.2, 'avg_grade': 0.2,
            'workload_rating': 0.2, 'general_rating': 0.2
        }
    if 'importance' not in session:
        session['importance'] = {
            'semantic': 3, 'credits': 3, 'avg_grade': 3,
            'workload_rating': 3, 'general_rating': 3
        }

    return render_template("filters.html", semester=session['filters']['semester'])


def split_summary_and_quotes(text):
    """Separates general summary text from bullet-point quotes."""
    if not text:
        return {'overview': '', 'quotes': []}

    lines = text.split('\n')
    overview_lines = []
    quotes = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if line looks like a bullet point (quote)
        if line.startswith(('*', '-', '•')):
            # Remove the bullet and surrounding quotes
            clean_quote = line.lstrip('*-• ').strip()
            if clean_quote.startswith('"') and clean_quote.endswith('"'):
                clean_quote = clean_quote[1:-1]
            quotes.append(clean_quote)
        else:
            # It is the objective summary text
            overview_lines.append(line)

    return {
        'overview': ' '.join(overview_lines),
        'quotes': quotes
    }


# STEP 4: Generate recommendations
@app.route("/recommendations")
def recommendations():
    # 1. Get all data from session
    completed_courses_data = session.get('completed_courses', [])
    filters_data = session.get('filters', {})
    weights = session.get('weights', {})
    user_query = session.get('user_query', '')

    semester = filters_data.get('semester', 'WINTER_2025_2026')
    no_exam = filters_data.get('no_exam', False)
    min_credits = filters_data.get('min_credits', 0)

    # Extract just the course IDs for the API
    completed_course_ids = [c['id'] for c in completed_courses_data]

    courses = []
    try:
        # 3. Call the recommendation engine
        ranked_df = recommend_courses(
            semester_name=semester,
            courses_list=completed_course_ids,
            user_query=user_query,
            no_exam=no_exam,
            min_credits=min_credits,
            semantic_weight=weights.get('semantic', 0.2),
            credits_weight=weights.get('credits', 0.2),
            avg_grade_weight=weights.get('avg_grade', 0.2),
            workload_rating_weight=weights.get('workload_rating', 0.2),
            general_rating_weight=weights.get('general_rating', 0.2)
        )
        # Convert to dicts
        courses = ranked_df.to_dict('records')

        # 4. Loop through and add the parsed fields to each course dictionary
        for course in courses:
            # Parse the raw summary string into Interest/Workload/BottomLine
            summary_parts = parse_review_summary(course.get('reviews_summary', ''))

            # SPLIT each part into 'overview' and 'quotes' for the UI
            course['summary_interest'] = split_summary_and_quotes(summary_parts['interest'])
            course['summary_workload'] = split_summary_and_quotes(summary_parts['workload'])
            course['summary_bottom_line'] = split_summary_and_quotes(summary_parts['bottom_line'])
            course['description'] = clean_description(course.get('description', ''))

    except Exception as e:
        print(f"Rec Error: {e}")
        flash(f"Error: {str(e)}")
        courses = []

    # Prepare display data so the HTML doesn't crash
    applied_filters = {
        "semester": semester,
        "no_exam": "Yes" if no_exam else "No",
        "min_credits": min_credits,
        "completed_count": len(completed_courses_data)
    }

    return render_template(
        "recommendations.html",
        courses=courses,
        filters=applied_filters,
        weights=weights,
        user_query=user_query
    )


# ============================================================================
# API ENDPOINTS FOR CHAT ASSISTANT (WITH SUPERVISOR)
# ============================================================================

@app.post("/api/chat")
def chat():
    """Handle chat messages - route through supervisor agent"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        agent_mode = data.get('agent_mode', None)  # 'rag' or 'rerank' or None
        conversation_history = data.get('history', [])

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Get current context from session
        filters = session.get('filters', {})
        weights = session.get('weights', {})
        user_query = session.get('user_query', '')
        semester = filters.get('semester', 'WINTER_2025_2026')

        # Build context for the supervisor
        context = {
            'semester': semester,
            'conversation_history': conversation_history,
            'current_recommendations': True,  # Indicates we have recommendations
            'filters': filters,
            'weights': weights,
            'user_query': user_query
        }

        # Route through supervisor
        result = supervisor_agent(
            user_message=user_message,
            agent_mode=agent_mode,
            context=context
        )

        return jsonify(result)

    except Exception as e:
        print(f"Chat error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'response': 'מצטער, אירעה שגיאה. אנא נסה שוב.',
            'sources': [],
            'success': False
        }), 500


@app.post("/api/rerank")
def api_rerank():
    """
    API endpoint to apply new ranking parameters and return updated courses
    Used by the reranker agent to dynamically update the page
    """
    try:
        data = request.get_json()
        new_weights = data.get('weights', {})
        new_filters = data.get('filters', {})
        new_query = data.get('query', '')

        # Update session with new parameters
        if new_weights:
            # Convert from supervisor format to session format
            session['weights'] = {
                'semantic': new_weights.get('semantic_weight', 0.2),
                'credits': new_weights.get('credits_weight', 0.2),
                'avg_grade': new_weights.get('avg_grade_weight', 0.2),
                'workload_rating': new_weights.get('workload_rating_weight', 0.2),
                'general_rating': new_weights.get('general_rating_weight', 0.2)
            }

        if new_filters:
            current_filters = session.get('filters', {})
            if 'no_exam' in new_filters:
                current_filters['no_exam'] = new_filters['no_exam']
            if 'min_credits' in new_filters:
                current_filters['min_credits'] = new_filters['min_credits']
            session['filters'] = current_filters

        if new_query is not None:  # Allow empty string
            session['user_query'] = new_query

        # Get updated data
        completed_courses_data = session.get('completed_courses', [])
        filters_data = session.get('filters', {})
        weights = session.get('weights', {})
        user_query = session.get('user_query', '')

        semester = filters_data.get('semester', 'WINTER_2025_2026')
        no_exam = filters_data.get('no_exam', False)
        min_credits = filters_data.get('min_credits', 0)
        completed_course_ids = [c['id'] for c in completed_courses_data]

        # Re-run recommendations
        ranked_df = recommend_courses(
            semester_name=semester,
            courses_list=completed_course_ids,
            user_query=user_query,
            no_exam=no_exam,
            min_credits=min_credits,
            semantic_weight=weights.get('semantic', 0.2),
            credits_weight=weights.get('credits', 0.2),
            avg_grade_weight=weights.get('avg_grade', 0.2),
            workload_rating_weight=weights.get('workload_rating', 0.2),
            general_rating_weight=weights.get('general_rating', 0.2)
        )

        courses = ranked_df.to_dict('records')

        # Add parsed summaries
        for course in courses:
            summary_parts = parse_review_summary(course.get('reviews_summary', ''))
            course['summary_interest'] = split_summary_and_quotes(summary_parts['interest'])
            course['summary_workload'] = split_summary_and_quotes(summary_parts['workload'])
            course['summary_bottom_line'] = split_summary_and_quotes(summary_parts['bottom_line'])
            course['description'] = clean_description(course.get('description', ''))

        return jsonify({
            'success': True,
            'courses': courses,
            'new_weights': weights,
            'new_query': user_query
        })

    except Exception as e:
        print(f"Rerank error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# WISHLIST LOGIC
# ============================================================================

# 1. Update Context Processor
@app.context_processor
def inject_wishlist():
    wishlist = session.get('wishlist', [])
    if not isinstance(wishlist, list): wishlist = []

    total_points = sum(float(item.get('points', 0) or 0) for item in wishlist)

    # Create a set of IDs for easy checking in templates
    wishlist_ids = {item['id'] for item in wishlist}

    return dict(
        wishlist=wishlist,
        wishlist_count=len(wishlist),
        wishlist_total=round(total_points, 1),
        wishlist_ids=wishlist_ids
    )


# 2. Add to Wishlist (Fixed is_xhr)
@app.route("/wishlist/add", methods=["POST"])
def add_to_wishlist():
    current_list = session.get('wishlist', [])
    if not isinstance(current_list, list): current_list = []

    course_id = request.form.get("course_id")
    course_name = request.form.get("course_name")
    try:
        points = float(request.form.get("course_points", 0))
    except:
        points = 0.0

    added = False
    if not any(c['id'] == course_id for c in current_list):
        current_list.append({'id': course_id, 'name': course_name, 'points': points})
        session['wishlist'] = current_list
        added = True

    # CHECK FOR AJAX/JSON REQUEST (FIXED)
    is_ajax = (request.headers.get('X-Requested-With') == 'XMLHttpRequest') or \
              (request.accept_mimetypes.best == 'application/json')

    if is_ajax:
        total = sum(float(i.get('points', 0)) for i in current_list)
        return jsonify({'status': 'success', 'count': len(current_list), 'total': total})

    if added: flash(f"Added {course_name}", "success")
    return redirect(request.referrer or url_for('index'))


# 3. Remove from Wishlist (Fixed is_xhr)
@app.route("/wishlist/remove", methods=["POST"])
def remove_from_wishlist():
    course_id = request.form.get("course_id")
    current_list = session.get('wishlist', [])
    if not isinstance(current_list, list): current_list = []

    session['wishlist'] = [c for c in current_list if c['id'] != course_id]

    # CHECK FOR AJAX/JSON REQUEST (FIXED)
    is_ajax = (request.headers.get('X-Requested-With') == 'XMLHttpRequest') or \
              (request.accept_mimetypes.best == 'application/json')

    if is_ajax:
        new_list = session['wishlist']
        total = sum(float(i.get('points', 0)) for i in new_list)
        return jsonify({'status': 'success', 'count': len(new_list), 'total': total})

    return redirect(request.referrer or url_for('index'))


# 4. Clear Wishlist
@app.route("/wishlist/clear", methods=["POST"])
def clear_wishlist():
    session['wishlist'] = []
    flash("Wishlist cleared.", "success")
    return redirect(request.referrer or url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)