import os
from dotenv import load_dotenv
from google import genai
import json
import re

# Import the existing RAG functionality
from src.agent import chat_with_assistant as rag_chat_with_assistant


def get_genai_client():
    """Initialize Google GenAI client"""
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOLGE_API_KEY")
    return genai.Client(api_key=GOOGLE_API_KEY)


def supervisor_agent(user_message, agent_mode=None, context=None):
    """
    Supervisor agent that routes requests to specialized agents
    
    Args:
        user_message: User's question or request
        agent_mode: 'rag' for Q&A, 'rerank' for personalization, or None for auto-detection
        context: Additional context (current recommendations, filters, etc.)
    
    Returns:
        dict with response, agent_used, and additional data
    """
    try:
        print(f"\n{'='*80}")
        print(f"🎯 SUPERVISOR AGENT ACTIVATED")
        print(f"{'='*80}")
        print(f"User message: {user_message}")
        print(f"Requested mode: {agent_mode}")
        print(f"{'='*80}\n")
        
        # If mode is specified, route directly
        if agent_mode == 'rag':
            return route_to_rag_agent(user_message, context)
        elif agent_mode == 'rerank':
            return route_to_reranker_agent(user_message, context)
        
        # Auto-detect mode (fallback if no button clicked)
        detected_mode = detect_intent(user_message)
        print(f"🤖 Auto-detected mode: {detected_mode}\n")
        
        if detected_mode == 'rerank':
            return route_to_reranker_agent(user_message, context)
        else:
            return route_to_rag_agent(user_message, context)
            
    except Exception as e:
        print(f"❌ Error in supervisor: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'response': f'מצטער, אירעה שגיאה: {str(e)}',
            'agent_used': 'error',
            'success': False
        }


def detect_intent(user_message):
    """
    Detect user intent from message (fallback for when no button is clicked)
    
    Returns:
        'rag' for questions about courses, 'rerank' for personalization requests
    """
    message_lower = user_message.lower()
    
    # Keywords for reranking
    rerank_keywords = [
        'דרג', 'סדר', 'שנה את', 'תעדיף', 'תמיין', 'rerank', 'sort', 'order',
        'העדף', 'יותר חשוב', 'פחות חשוב', 'בראש', 'בסוף', 'personalize',
        'התאם אישית', 'שנה משקל', 'שנה דירוג'
    ]
    
    if any(keyword in message_lower for keyword in rerank_keywords):
        return 'rerank'
    
    return 'rag'  # Default to RAG for questions


def route_to_rag_agent(user_message, context):
    """
    Route to the RAG Q&A agent for course questions
    """
    print(f"\n📚 Routing to RAG Agent...")
    
    semester = context.get('semester', 'WINTER_2025_2026') if context else 'WINTER_2025_2026'
    semester_rag = f"{semester}_RAG"
    conversation_history = context.get('conversation_history', []) if context else []
    
    # Call the existing RAG agent
    result = rag_chat_with_assistant(
        user_message=user_message,
        semester_name=semester_rag,
        conversation_history=conversation_history
    )
    
    # Add agent metadata
    result['agent_used'] = 'rag'
    result['action_type'] = 'chat'
    
    return result


def route_to_reranker_agent(user_message, context):
    """
    Route to the reranker agent for personalized ranking
    """
    print(f"\n🎨 Routing to Reranker Agent...")
    
    if not context or 'current_recommendations' not in context:
        return {
            'response': 'לא מצאתי המלצות פעילות. אנא וודא שיש לך המלצות מוצגות בעמוד.',
            'agent_used': 'reranker',
            'success': False,
            'action_type': 'chat'
        }
    
    # Get current state
    current_filters = context.get('filters', {})
    current_weights = context.get('weights', {})
    user_query = context.get('user_query', '')
    
    # Use LLM to understand the reranking request
    rerank_result = analyze_rerank_request(
        user_message=user_message,
        current_filters=current_filters,
        current_weights=current_weights,
        current_query=user_query
    )
    
    if rerank_result['success']:
        return {
            'response': rerank_result['explanation'],
            'agent_used': 'reranker',
            'success': True,
            'action_type': 'rerank',
            'new_weights': rerank_result['new_weights'],
            'new_filters': rerank_result.get('new_filters'),
            'new_query': rerank_result.get('new_query')
        }
    else:
        return {
            'response': rerank_result.get('explanation', 'לא הצלחתי להבין את הבקשה. אנא נסה שוב.'),
            'agent_used': 'reranker',
            'success': False,
            'action_type': 'chat'
        }


def analyze_rerank_request(user_message, current_filters, current_weights, current_query):
    """
    Use LLM to analyze the reranking request and determine new weights/filters
    
    Returns:
        dict with new_weights, new_filters, new_query, explanation
    """
    load_dotenv()
    CHAT_MODEL = os.getenv("CHAT_MODEL")
    client = get_genai_client()
    
    system_prompt = """אתה עוזר שמפענח בקשות של סטודנטים לשינוי דירוג קורסים.
המשימה שלך היא להבין מה הסטודנט רוצה ולהחזיר הגדרות חדשות במבנה JSON.

הפרמטרים שאתה יכול לשנות:
1. **weights** (משקלים, סכום חייב להיות 1.0):
   - semantic_weight: התאמה לשאילתת החיפוש (0-1)
   - credits_weight: מספר נקודות (0-1)
   - avg_grade_weight: ציון ממוצע גבוה (0-1)
   - workload_rating_weight: עומס נמוך (0-1)
   - general_rating_weight: דירוג כללי גבוה (0-1)

2. **query** (שאילתת חיפוש טקסטואלית):
   - השאילתה משפיעה על semantic_weight
   - דוגמאות: "קורסים עם פייתון", "למידת מכונה", "קורסים קלים"

3. **filters**:
   - no_exam: true/false (האם לסנן קורסים ללא מבחן)
   - min_credits: מספר (מינימום נקודות)

דוגמאות:
- "אני רוצה קורסים קלים" → העלה workload_rating_weight
- "קורסים עם פייתון" → עדכן query ל-"Python programming"
- "דרג לפי ציונים" → העלה avg_grade_weight
- "רק קורסים ללא מבחן" → no_exam: true
- "העדף קורסים במידת מידע" → עדכן query ל-"מדעי המידע כריית נתונים"

חזור **רק** JSON בפורמט הבא (ללא markdown backticks):
{
  "new_weights": {
    "semantic_weight": 0.2,
    "credits_weight": 0.0,
    "avg_grade_weight": 0.3,
    "workload_rating_weight": 0.3,
    "general_rating_weight": 0.2
  },
  "new_query": "טקסט חיפוש או null",
  "new_filters": {
    "no_exam": true/false,
    "min_credits": number
  },
  "explanation": "הסבר קצר בעברית מה שינית ולמה (1-2 משפטים)"
}

אם לא הצלחת להבין את הבקשה, החזר:
{
  "success": false,
  "explanation": "לא הבנתי את הבקשה. אנא נסח מחדש."
}"""

    user_prompt = f"""בקשת הסטודנט: "{user_message}"

הגדרות נוכחיות:
- משקלים: {json.dumps(current_weights, ensure_ascii=False)}
- שאילתה: "{current_query}"
- פילטרים: {json.dumps(current_filters, ensure_ascii=False)}

מה הגדרות חדשות אתה ממליץ?"""

    try:
        print(f"\n🤖 Calling LLM to analyze rerank request...")
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=user_prompt,
            config={
                "system_instruction": system_prompt,
                "temperature": 0.3,
                "max_output_tokens": 1000,
            }
        )
        
        response_text = response.text.strip()
        print(f"Raw LLM response: {response_text}")
        
        # Clean response (remove markdown if present)
        response_text = re.sub(r'^```json\s*', '', response_text)
        response_text = re.sub(r'\s*```$', '', response_text)
        
        result = json.loads(response_text)
        
        # Validate weights sum to 1.0
        if 'new_weights' in result:
            weights = result['new_weights']
            total = sum(weights.values())
            if abs(total - 1.0) > 0.01:  # Allow small floating point errors
                # Normalize
                for key in weights:
                    weights[key] = round(weights[key] / total, 3)
        
        result['success'] = True
        print(f"✅ Parsed rerank request successfully")
        return result
        
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON: {e}")
        return {
            'success': False,
            'explanation': 'מצטער, לא הצלחתי לפענח את הבקשה. אנא נסח אותה בצורה פשוטה יותר.'
        }
    except Exception as e:
        print(f"❌ Error in analyze_rerank_request: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'explanation': f'אירעה שגיאה: {str(e)}'
        }


# Testing
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 TESTING SUPERVISOR AGENT")
    print("="*80 + "\n")
    
    # Test 1: RAG mode
    print("\n🔬 TEST 1: RAG Agent (Q&A)")
    result1 = supervisor_agent(
        "מה אומרים על עומס העבודה בקורסי כלכלה?",
        agent_mode='rag',
        context={'semester': 'WINTER_2025_2026'}
    )
    print(f"Agent used: {result1['agent_used']}")
    print(f"Response: {result1['response'][:200]}...")
    
    # Test 2: Reranker mode
    print("\n🔬 TEST 2: Reranker Agent")
    result2 = supervisor_agent(
        "אני רוצה קורסים קלים עם ציונים גבוהים",
        agent_mode='rerank',
        context={
            'current_recommendations': True,
            'filters': {'semester': 'WINTER_2025_2026', 'no_exam': False, 'min_credits': 0},
            'weights': {
                'semantic_weight': 0.2,
                'credits_weight': 0.2,
                'avg_grade_weight': 0.2,
                'workload_rating_weight': 0.2,
                'general_rating_weight': 0.2
            },
            'user_query': ''
        }
    )
    print(f"Agent used: {result2['agent_used']}")
    print(f"Action type: {result2.get('action_type')}")
    print(f"Response: {result2['response']}")
    if result2.get('new_weights'):
        print(f"New weights: {result2['new_weights']}")
