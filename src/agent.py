import os
from dotenv import load_dotenv
from pinecone import Pinecone
from google import genai
import json
from google.genai import types
from src.knowledgebase import embed_query
# Initialize Google GenAI client




def get_pinecone():
    """Initialize Pinecone client"""
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError("No API key found. Please check your .env file.")

    pc = Pinecone(api_key=api_key)
    return pc


def get_index_by_semester(semester_name):
    """Get Pinecone index for specific semester"""
    pc = get_pinecone()
    kb_name = os.getenv(semester_name)

    if not kb_name:
        raise ValueError(f"No index found for semester: {semester_name}")

    index = pc.Index(host=kb_name)
    return index





def search_reviews(query, semester_name="WINTER_2025_2026_RAG", top_k=15):
    """
    Search for relevant course reviews based on user query

    Args:
        query: User's question
        semester_name: Semester identifier for the index
        top_k: Number of results to return

    Returns:
        List of relevant course reviews with metadata
    """
    try:
        print(f"\n{'=' * 80}")
        print(f"🔍 SEARCHING REVIEWS")
        print(f"{'=' * 80}")
        print(f"Query: {query}")
        print(f"Semester: {semester_name}")
        print(f"Top K: {top_k}")

        # Get index
        index = get_index_by_semester(semester_name)
        print(f"✅ Connected to index: {semester_name}")

        # Generate query embedding
        query_embedding = embed_query(query)

        if query_embedding is None:
            print("❌ Failed to generate embedding")
            return []

        print(f"✅ Generated embedding (dim: {len(query_embedding)})")

        # Search in Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        print(f"\n📊 SEARCH RESULTS: Found {len(results.matches)} matches")
        print(f"{'-' * 80}")

        # Extract relevant information
        context_chunks = []
        for i, match in enumerate(results.matches, 1):
            course_id = match.metadata.get('course_id', 'N/A')
            course_title = match.metadata.get('title', 'N/A')
            review_text = match.metadata.get('chunk_text', '')
            score = match.score

            print(f"\n[{i}] Score: {score:.4f}")
            print(f"    Course: {course_title} ({course_id})")
            print(f"    Review preview: {review_text[:150]}...")

            context_chunks.append({
                'course_id': course_id,
                'course_title': course_title,
                'review_text': review_text,
                'score': score
            })

        print(f"\n{'=' * 80}\n")
        return context_chunks

    except Exception as e:
        print(f"❌ Error searching reviews: {e}")
        import traceback
        traceback.print_exc()
        return []


def build_context(search_results):
    """Build context string from search results for the LLM"""
    if not search_results:
        print("⚠️ No search results to build context from")
        return "לא נמצאו ביקורות רלוונטיות."

    context_parts = []

    for i, result in enumerate(search_results, 1):
        course_title = result['course_title']
        course_id = result['course_id']
        review = result['review_text']



        context_parts.append(f"\n--- ביקורת {i} | {course_title} ({course_id}) ---\n{review}")

    full_context = "\n".join(context_parts)

    print(f"\n{'=' * 80}")
    print(f"📝 BUILT CONTEXT FOR LLM")
    print(f"{'=' * 80}")
    print(f"Total context length: {len(full_context)} characters")
    print(f"Number of reviews: {len(search_results)}")
    print(f"\nContext preview (first 500 chars):")
    print(full_context[:500])
    print(f"{'=' * 80}\n")

    return full_context


def chat_with_assistant(user_message, semester_name="WINTER_2025_2026_RAG", conversation_history=None):
    """
    Main chat function - handles user queries using RAG

    Args:
        user_message: User's question
        semester_name: Semester identifier
        conversation_history: Previous conversation (optional)

    Returns:
        dict with 'response' and 'sources'
    """
    load_dotenv()

    # 2. Get the key securely
    CHAT_MODEL = os.getenv("CHAT_MODEL")
    GOOGLE_API_KEY = os.getenv("GOOLGE_API_KEY")
    genai_client = genai.Client(api_key=GOOGLE_API_KEY)
    try:
        print(f"\n{'#' * 80}")
        print(f"💬 NEW CHAT REQUEST")
        print(f"{'#' * 80}")
        print(f"User message: {user_message}")
        print(f"Semester: {semester_name}")
        print(f"Conversation history length: {len(conversation_history) if conversation_history else 0}")

        # Search for relevant reviews
        search_results = search_reviews(user_message, semester_name, top_k=15)

        # Build context from search results
        context = build_context(search_results)

        # Build conversation history
        if conversation_history is None:
            conversation_history = []

        # System prompt focused on answering questions
        system_prompt = """אתה עוזר וירטואלי של CheeseSpoon - מערכת המלצות קורסים של הטכניון.
התפקיד שלך הוא לענות על שאלות ספציפיות של סטודנטים על קורסים, בהתבסס על ביקורות של סטודנטים שלמדו את הקורסים.

הנחיות חשובות:
1. ענה בעברית בצורה ישירה ומדויקת על השאלה שנשאלה
2. התבסס אך ורק על המידע שניתן לך מהביקורות - אל תמציא מידע
3. אם השאלה היא על נושא ספציפי (למשל: "האם יש שיעורי בית?", "מה אומרים על המרצה?", "כמה זמן לוקח להכין למבחן?") - חפש את המידע הרלוונטי בביקורות וענה באופן ממוקד
4. כשמשווים בין קורסים - הצג את ההבדלים הספציפיים שנשאלו (עומס, קושי, איכות הוראה וכו')
5. אם אין מידע על הנושא הספציפי בביקורות - אמר זאת בכנות: "לא מצאתי מידע על נושא זה בביקורות"
6. תן תשובה תמציתית אבל מלאה - אל תסכם כללי אלא ענה על השאלה הקונקרטית
7. אם יש דעות שונות בביקורות - הצג את מגוון הדעות

דוגמאות לסוג השאלות שאתה צריך לענות עליהן:
- "מה אומרים על עומס העבודה בקורס X?"
- "האם יש מבחן או שזה פרויקט?"
- "מה הסטודנטים אומרים על המרצה Y?"
- "כמה קשה הקורס הזה?"
- "האם כדאי לקחת את הקורס X או Y?"
- "מה צריך לדעת מראש בשביל הקורס?"
- "איך נראה המבחן?"

זכור: אתה לא מסכם את הקורס - אתה עונה על שאלות ספציפיות!"""

        # Build the prompt
        user_prompt = f"""שאלת הסטודנט: {user_message}

ביקורות רלוונטיות מהמאגר:
{context}

בבקשה ענה על השאלה בהתבסס על הביקורות. 
אם השאלה משווה בין קורסים - הצג את ההבדלים הספציפיים.
אם אין מידע רלוונטי - אמר זאת."""

        print(f"\n{'=' * 80}")
        print(f"🤖 CALLING LLM")
        print(f"{'=' * 80}")
        print(f"Model: {CHAT_MODEL}")
        print(f"Temperature: 0.4")
        print(f"Max tokens: 800")
        print(f"\nFull prompt (first 800 chars):")
        print(user_prompt[:800])
        print(f"{'=' * 80}\n")

        # Call Google GenAI API
        response = genai_client.models.generate_content(
            model=CHAT_MODEL,
            contents=user_prompt,
            config={
                "system_instruction": system_prompt,
                "temperature": 0.4,  # Lower temperature for more focused answers
                "max_output_tokens": 5000,
            }
        )

        assistant_response = response.text

        print(f"\n{'=' * 80}")
        print(f"✅ LLM RESPONSE RECEIVED")
        print(f"{'=' * 80}")
        print(f"Response length: {len(assistant_response)} characters")
        print(f"Response preview (first 300 chars):")
        print(assistant_response[:300])
        print(f"{'=' * 80}\n")

        # Prepare sources for citation
        sources = []
        seen_courses = {}

        for result in search_results:
            course_key = f"{result['course_id']}_{result['course_title']}"

            # Only show each course once, with highest relevance score
            if course_key not in seen_courses:
                seen_courses[course_key] = {
                    'course_id': result['course_id'],
                    'course_title': result['course_title'],

                    'relevance_score': round(result['score'] * 100, 1)
                }

        # Take top 4 unique courses
        sources = list(seen_courses.values())[:4]

        print(f"📚 SOURCES PREPARED: {len(sources)} unique courses")
        for i, source in enumerate(sources, 1):
            print(f"  [{i}] {source['course_title']} ({source['course_id']}) - {source['relevance_score']}%")

        print(f"\n{'#' * 80}\n")

        return {
            'response': assistant_response,
            'sources': sources,
            'success': True
        }

    except Exception as e:
        print(f"\n❌ ERROR IN CHAT_WITH_ASSISTANT")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'#' * 80}\n")

        return {
            'response': f"מצטער, אירעה שגיאה: {str(e)}",
            'sources': [],
            'success': False
        }


def answer_course_question(course_id, question, semester_name="WINTER_2025_2026_RAG"):
    """
    Answer a specific question about a particular course

    Args:
        course_id: Course ID
        question: Specific question about the course
        semester_name: Semester identifier

    Returns:
        Answer text
    """
    try:
        print(f"\n{'=' * 80}")
        print(f"❓ COURSE-SPECIFIC QUESTION")
        print(f"{'=' * 80}")
        print(f"Course ID: {course_id}")
        print(f"Question: {question}")

        # Create targeted query
        query = f"קורס {course_id} {question}"
        print(f"Formatted query: {query}")

        # Search for reviews of this specific course
        search_results = search_reviews(query, semester_name, top_k=15)

        # Filter to only this course and high relevance
        course_reviews = [
            r for r in search_results
            if r['course_id'] == str(course_id) and r['score'] > 0.3
        ]

        print(f"Filtered to {len(course_reviews)} reviews for course {course_id}")

        if not course_reviews:
            print("⚠️ No relevant reviews found")
            return "לא נמצא מידע ספציפי על נושא זה בביקורות."

        # Build context
        context = build_context(course_reviews)

        # Create focused prompt
        prompt = f"""שאלה על קורס {course_id}: {question}

ביקורות רלוונטיות:
{context}

ענה על השאלה בצורה ישירה וספציפית בהתבסס על הביקורות.
אל תסכם את כל הקורס - רק ענה על השאלה הספציפית שנשאלה.
אם אין מידע - אמר "לא מצאתי מידע על כך בביקורות"."""

        print(f"\n🤖 Calling LLM for course-specific answer...")

        response = genai_client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 400,
            }
        )

        answer = response.text
        print(f"✅ Answer received: {answer[:200]}...")
        print(f"{'=' * 80}\n")

        return answer

    except Exception as e:
        print(f"❌ Error answering course question: {e}")
        import traceback
        traceback.print_exc()
        return "לא ניתן לענות על השאלה כרגע."


# For testing
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🧪 TESTING RAG Q&A SYSTEM")
    print("=" * 80 + "\n")

    # Test 1: Specific question about workload
    print("\n" + "🔬 TEST 1: Question about workload")
    result1 = chat_with_assistant("מה אומרים על עומס העבודה בקורסי כלכלה?")
    print("\n📋 FINAL RESULT:")
    print("Response:", result1['response'])
    print("Sources:", result1['sources'])
    print("\n" + "=" * 80 + "\n")

    # Test 2: Question about exams
    print("\n" + "🔬 TEST 2: Question about exams")
    result2 = chat_with_assistant("איך נראים המבחנים בקורסים עם פרץ חובב?")
    print("\n📋 FINAL RESULT:")
    print("Response:", result2['response'])
    print("Sources:", result2['sources'])
    print("\n" + "=" * 80 + "\n")