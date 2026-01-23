import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from typing import List, Dict, Optional
import warnings

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
# Alternative lightweight models:
# "HuggingFaceTB/SmolLM2-1.7B-Instruct" - good multilingual support
# "Qwen/Qwen2.5-1.5B-Instruct" - excellent for Hebrew

USE_QUANTIZATION = False  # Set to False for 1B models (they're small enough)

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

REVIEW_SUMMARY_PROMPT = """אתה עוזר אקדמי שמנתח ביקורות של סטודנטים על קורסים.



משימתך:

1. בחר עד 4 ציטוטים קצרים ומייצגים מהביקורות שמתארים את הקורס באופן כללי (לא ספציפית על מרצה או מתרגל ספציפי).

2. תן עדיפות לביקורות שנראות עדכניות יותר (מופיעות בתחילת הרשימה).

3. כתוב סיכום של 2-3 שורות על החוויה הכללית של הסטודנטים - על מה להיזהר, מה חשוב לדעת.


דוגמא:
הביקורות:
שם המרצה: אסף
חוות דעת - הרצאות: כל שבוע הרצאה עם נושא חדש, מדברים על נושאים ודנים בהם, מרענן את השבוע מאלגברה ודיסקרטית. הרצאות שמדברות תכלס על התפקיד בתעשייה, אחלה לסמסטר ראשון.

שם המתרגל/ת: נחמיה
חוות דעת - תרגולים: אין על נחמיה, חבל שלא יצא לנו ללמוד איתו בלייב.

שעורי הבית: כל שבוע עבודה קלילה בזוגות, בסוף הסמסטר מעבדת רובוטיקה (גם בזום).
* את הציון של העבודות תהיו מוכנים לקבל אחרי הציון של המבחן

המבחן: הזכיר לי נשכחות של מבחנים מימי התיכון

השורה התחתונה: אחלה קורס, מעלה ממוצע לסמסטר ראשון, ומעשיר את הידע לגבי התחומים של מהנדס תעשייה בעתיד
כללי
עומס
---
שם המרצה: אסף
חוות דעת - הרצאות: כל שבוע מדברים על נושא אחר. ממש נהניתי בהרצאות! אחלה הפוגה משאר הקורסים היותר מתמטיים במהלך השבוע. אסף מרצה מעולה ורואים שאכפת לו מהקורס ושהוא אוהב את התחום.

שם המתרגל/ת: נחמיה
חוות דעת - תרגולים: לא הלכתי לתרגולים האמת, הספיק לי לעבור על המצגת והפתרונות בבית.

שעורי הבית: גיליון להגשה כל שבוע בזוגות/יחידים. לא קשה במיוחד :)

המבחן: אני כן חושבת שהיה מבחן הוגן מבחינת רוב השאלות שנשאלו בו והם הצליחו לכסות כמעט את כל החומר הנלמד. עם זאת, היו לא מעט סעיפים טריקיים ומבלבלים ועיקר הבעיה היה הזמן. פשוט יותר מדי שאלות וסעיפים למבחן של שעתיים כך שהרבה אנשים אפילו לא הספיקו חלק מהמבחן. אפשר להביא דף נוסחאות אישי שמכינים לבד אז גם ההכנה שלו עוזרת לעבור על החומר.

השורה התחתונה: אחלה קורס לסמסטר ראשון. תעברו על המצגות לפני המבחן ועל שאלות ממבחני עבר ותוסיפו לדף נוסחאות כל דבר שנראה לכם רלוונטי ותהיו בסדר :)
כללי
עומס
---
שם המרצה: אסף
חוות דעת - הרצאות: נוכחות חובה

שם המתרגל/ת: נחמיה
חוות דעת - תרגולים: לא הכרחי להבנת החומר

שעורי הבית: תואם לרמת המבחן

המבחן: כל החומר הנלמד בתרגולים הצליח להכנס למבחן של שעתיים אז כדאי לעקוב אחרי מה לומדים במהלך הסמסטר

השורה התחתונה: אפשרי וסביר, ניתן ללמוד גם בלי להגיע להרצאות או לתרגולים פרט לעניין הנוכחות החובה
כללי
עומס
---
שם המרצה: אסף (איירון מן) אברהמי
חוות דעת - הרצאות: תתכוננו לזה שרוב הזמן אסף מדבר על עצמו ועל כמה הוא תותח. את החומר הוא לרוב מעביר מהמצגת. יש שאלת שבוע כל שבוע ככה שלפעמים זה כן מתפתח לשיח נחמד, תלוי כמה הוא נותן לזה מקום וזמן. הנוכחות חובה אז אין יותר מדי ברירה. לנו זה היה פתיחת שבוע חמודה ולא מעבר. תשתדלו כן להשתתף בשביל העוד כמה נקודות האלה בסוף הקורס (ושיזכור את השם שלכם), ותעשו את המצגת 5 דק' שמוסיפה נקודות חינם לציון הסופי.

שם המתרגל/ת: נחמיה ירון
חוות דעת - תרגולים: נחמיה לא באמת יודע ללמד, עד היום לא ברור לי מה ההשכלה/המקצוע שלו. כדי למלא את כל השעה שיש לו הוא פשוט פותר את השאלה מהמצגת על הלוח (מה שמופיע בשקופית הבאה אשכרה), וגם יש במודל פתרונות לתרגולים שלו כקובץ ולא רק כמצגת. התרגולים אצלנו לא היו חובה וגם הם תקעו אותם ב6 בערב אז אף אחד לא הלך. בקיצור- מיותר.

שעורי הבית: כל שבוע, ממוחזרים רצח. אפשר לרפרנס אבל אפשר בכיף גם לפתור אותם לבד (או בזוגות) כדי לוודא הבנה של החומר השבוע (כאמור- כל שבוע נושא אחר לגמרי בקורס). תתכוננו נפשית לזה שהוא לא מחזיר ציונים על המטלות עד המבחן (תנסו להציק לו ולאסף על זה במהלך הסמסטר יותר).

המבחן: בעיקרון אמור להיות מבחן מתנה אם לומדים אליו באמת. לא לזלזל כי זה באסה לקבל ציון גרוע (או לא לעבור) מבחן שיכול ללכת סבבה לגמרי, לעומת השאר בסמסטר הזה. מה שכן- אצלנו הם יצאו מניאקים קצת. הם ניסו לדחוס את כל החומר של הקורס במבחן אחד ככה שכל סעיף לקח נצח ואנשים לא הספיקו לענות על כל המבחן (שעתיים). במבחני עבר זה לא היה ככה (וגם הם לא הסכימו לפרסם את המבחן של שנה שעברה והפתרון). וגם היו בעיות בניסוחים. סהכ כל הבעיות התנקזו ל5 נק פקטור שטרחו לתת, מקווה שיהיו יותר הוגנים להבא.

השורה התחתונה: החומר עצמו לא מאוד קשה, אבל כן מורכב מהרבה נושאים שכל אחד עומד בפני עצמו וכדאי לשים לב לכולם. סהכ קורס חביב, נותן סיפתח (קלוש אבל נותן) לתואר. תתעקשו על דברים שמגיע לכם לקבל ותזרמו עם ההנפצות שלהם (סיור מיותר, סדנת הדפסת תלת מימד, עבודה על התוכנה שאסף המציא וכו')
כללי
עומס
---
שם המרצה: פרופסור אסף אברהמי
חוות דעת - הרצאות: אסף מדבר על עצמו בקטע שכבר גורם אי נוחות

שם המתרגל/ת: נחמיה ירון
חוות דעת - תרגולים:
חחחחח
שעורי הבית:
תקוע בין כל שאר המטלות האמיתיות, לא כזה קריטי להתעמק במהלך הסמסטר אפשר להעזר ברפרנסים.
המבחן:
לקחת 4 ימים לפני לחרוש רק על זה ואז מוציאים ציון טוב
השורה התחתונה: קורס מבאס אין מה לעשות. הנושאים עצמם סופר מענינים ובהתחלה של כל נושא יש פוטנציאל שבאמת יהיה מעניין, אבל ... אסף ונחמיה... זה לא קורס אמיתי ותנסו להתכונן נפשית לכל הדברים שהם תוקעים במהלך הסמטסר(חשבשבת, סדנה,פרויקטון)
כללי
עומס
---
שם המרצה: פרופ' מנכ"ל קצין בדימוס וגיבור המולדת אסף אברהמי
חוות דעת - הרצאות: תכלס, חוץ מזה שהוא מתנשא, זה דיי אחלה, צריך להקשיב כי יש שאלה על מה שהוא מדבר במבחן. לפעמים יש הרצאות אורח שיכולות להיות אחלה, הדברים באמצע (תכ"ן וחשבשבת) חרטא לגמרי. בסוף חמוד, אמור להעלות ציון בסמסטר א'

שם המתרגל/ת: נחמיה "זה אינטואיטיבי" ירון
חוות דעת - תרגולים: חמוד, בסוף זה החלק האמיתי של הציון, מומלץ לראות את המצגות ואם לא הבנתם לבוא.

שעורי הבית: חרטא, פעם בשבוע שעה גג בזוגות

המבחן: הוגן, כאילו החומר לפעמים ממש לא מרגיש חשוב אבל צריך להבין את מה שרוצים ממך.

השורה התחתונה: אין בחירה אז בואו, בראשון זו אחלה דרך לפתוח שבוע בכיף
כללי
עומס
---
שם המרצה: אסף אברהמי
חוות דעת - הרצאות: נוכחות חובה. בגדול נחמד לפתוח ככה את יום ראשון ולא בקורסים של מתמטיקה. אסף מת על עצמו ובעיקר מדבר עליו ועל החברה שלו בהרצאות.

שם המתרגל/ת: נחמיה ירון
חוות דעת - תרגולים: מפסיקים ללכת אחרי התרגולים הראשונים, לא חובה

שעורי הבית: העתק הדבק מהמצגת של התרגול, אבל מעיק שיש שיעורי בית בזה פעם בשבוע כי לפעמים זה גוזל זמן.

המבחן: המבחן השנה היה ארוך וקשה מאוד ביחס לשנים קודמות. לא היה הוגן.

השורה התחתונה: קורס חמוד, לא צריך להשקיע יותר מידי חוץ מהשיעורי בית כל שבוע.
כללי
עומס
---
שם המרצה: אסף
חוות דעת - הרצאות:
מלא פוטנציאל להיות מעניין וכייפי אבל אסף כלכך אוהב את התחת של עצמו שזה די הורס את השיעור. היה יכול להיות אחלה הזדמנות לפתח שיחה מעניינת בכל מני נושאים (כל שבוע יש שאלת שבוע ופותחים עליה דיון) אבל מרגיש שאין מקום אמיתי לשתף בכלום. הנוכחות היא חובה אבל זה חרטא זה רק כי בסוף הסמסטר אסף מביא ציון על ההשתתפות בכיתה. הייתי ממליצה פשוט לדבר מספיק כדי שיזכור את השם שלכם ואז מקבלים מאה
בנוסף יש מצגת שלא חייבים לעשות אבל מעלה את הציון הסופי אז למה לא

שם המתרגל/ת: נחמיה
חוות דעת - תרגולים:
לא הלכתי היה מיותר

שעורי הבית:
יש כל שבוע, ובסוף זה 15% מהציון. ממחוזרים לגמרי, אפשר לעשות 99 אחוז עם רפרנסים אבל גם ממש קל לעשות לבד. הם תכלס להגשה בזוגות אבל לא חובה. לנו הסמסטר הייתה בודקת (לא נחמיה) אז דווקא החזירו ציונים די מהר.
השיעורי בית המעצבים באמת זה העבודה על החשבשבת והפרוייקטון. הייתי ממליצה להתחיל את הפרוייקטון כמה שיותר מוקדם (אפילו שבוע 3-4) ולא להשאיר את זה לסוף הסמסטר כשכבר יש עומס ומבחנים על הראש

המבחן:
היה עמוס ממש ולא אפשרי לסיים בשעתיים, נתנו פקטור של 4 נקודות שבאמת היה בדיחה יחסית לממוצע. בתכלס החומר לא מסובך (חוץ מהשבוע של הסתברות שהוא חרא ותמיד מופיע במבחן) אבל פשוט יש מלא חומר, ומותר רק דף נוסחאות אחד אז צריך לכתוב ממש קטן כי באמת יש מלא מידע שצריך למבחן.

השורה התחתונה: קורס חובה חמוד, ואחלה הזדמנות להכיר את האנשים שאיתכם בתואר. המבחן קצת מבאס
כללי
עומס
---
שם המרצה: אסף אברהמי
חוות דעת - הרצאות: נוכחות חובה אבל נחמד לפתוח את השבוע במשהו שהוא לא מתמטיקה. אסף בעיקר אוהב לדבר על עצמו ועל החברה שלו ומכניס את זה לדיון בכל הזדמנות. מעבר לזה ההרצאות נחמדות, לא מעבר. הרצאות האורח של נועם היו אחלה לגמרי.

שם המתרגל/ת: נחמיה ירון
חוות דעת - תרגולים: אחרי שני תרגולים הבנו שהם מיותרים והפסקנו ללכת.

שעורי הבית: הרוב זה העתק הדבק מהמצגת של התרגול רק שמשנים נתונים. אבל שזה לא כמו במצגת זה שבירת ראש. יש גיליון כל שבוע שזה מעיק כי לפעמים יוצא שמקדישים הרבה זמן לקורס יחסית קטן יותר ביחס לשאר.

המבחן: המבחן הפעם היה ארוך ממש ביחס לשנים קודמות ולדעתי גם קשה הרבה יותר. מבאס כי הקורס אמור להעלות את הממוצע אבל בגדול אין מה לדאוג מלעבור את המבחן.

השורה התחתונה: קורס חמוד לסמסטר ראשון בתעשייה וניהול
כללי
עומס

פורמט התשובה שלך:
ציטוטים:
"כל שבוע עבודה קלילה בזוגות"
"כל החומר הנלמד בתרגולים הצליח להכנס למבחן של שעתיים אז כדאי לעקוב אחרי מה לומדים במהלך הסמסטר"
"אפשר להביא דף נוסחאות אישי שמכינים לבד"


סיכום:
הסטודנטים מתארים את הקורס כהפוגה קלילה ונחמדה מהעומס המתמטי של סמסטר א', אם כי רבים ציינו שהמרצה נוטה להאדיר את עצמו והתרגולים לרוב מרגישים מיותרים. מבחינת עומס, ישנן מטלות שבועיות (לרוב טכניות/ממוחזרות), אך שימו לב שהמבחן האחרון תואר כעמוס וארוך מאוד ביחס לזמן המוקצב, בניגוד לשנים עברו.
"""

TOPIC_EXTRACTION_PROMPT = """You are an expert in analyzing academic curricula.
First, translate the hebrew text to english and then:
Your Task: Extract the main topics taught in the course from the official description provided below.

Course Description:
{description}

Return a list of topics in JSON format only, with no additional text or markdown formatting:
{{"topics": ["Topic 1", "Topic 2", "Topic 3"]}}

Requirements:
1. Use English names for the topics.
2. Return between 2 to 6 topics.
"""


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    """Load the LLM model with optimal settings for Colab GPU"""
    print("Loading model...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=HUGGING_FACE_TOKEN if HUGGING_FACE_TOKEN != "YOUR_HF_TOKEN_HERE" else None
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model based on quantization availability
    if USE_QUANTIZATION and QUANTIZATION_AVAILABLE:
        print("Loading with 4-bit quantization...")
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            token=HUGGING_FACE_TOKEN if HUGGING_FACE_TOKEN != "YOUR_HF_TOKEN_HERE" else None,
            trust_remote_code=True
        )
    else:
        print("Loading model in standard mode...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            token=HUGGING_FACE_TOKEN if HUGGING_FACE_TOKEN != "YOUR_HF_TOKEN_HERE" else None,
            trust_remote_code=True
        )

    print(f"✓ Model loaded successfully on {model.device}")
    print(f"  Model size: ~{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

    return model, tokenizer


# ============================================================================
# LLM INFERENCE FUNCTIONS
# ============================================================================

def generate_response(model, tokenizer, prompt: str, max_tokens: int = 512) -> str:
    """Generate response from the model"""

    messages = [
        {"role": "system", "content": "You are a native Hebrew and English speaker."},
        {"role": "user", "content": prompt}
    ]

    # Format prompt for the model
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except:
        # Fallback if chat template not available
        formatted_prompt = f"<|system|>אתה עוזר מועיל ומדויק.</s>\n<|user|>{prompt}</s>\n<|assistant|>"

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,  # Lower temperature for more consistent outputs
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def extract_review_summary(model, tokenizer, reviews: str) -> str:
    """Extract quotes and summary from student reviews"""

    if pd.isna(reviews) or not reviews.strip():
        return ""

    # Truncate reviews if too long (keep first 2000 chars for 1B model)
    reviews_text = str(reviews)[:2000]

    prompt = REVIEW_SUMMARY_PROMPT.format(reviews=reviews_text)

    print("\n" + "=" * 80)
    print("PROCESSING REVIEW SUMMARY")
    print("=" * 80)

    try:
        response = generate_response(model, tokenizer, prompt, max_tokens=400)

        print(f"LLM Output:\n{response}")
        print("=" * 80)

        # Parse the response
        summary = parse_review_response(response)
        return summary
    except Exception as e:
        print(f"❌ Error: {e}")
        return ""


def parse_review_response(response: str) -> str:
    """Parse the LLM response to extract formatted summary"""

    # Try to extract quotes and summary sections
    quotes = []
    summary = ""

    # Extract quotes - look for text in quotation marks
    quote_pattern = r'"([^"]+)"'
    found_quotes = re.findall(quote_pattern, response)
    quotes = found_quotes[:4]  # Maximum 4 quotes

    # Extract summary
    if "SUMMARY:" in response:
        summary_part = response.split("SUMMARY:")[-1].strip()
        # Take first paragraph after SUMMARY:
        summary = summary_part.split("\n\n")[0].strip()
        # Remove any remaining quotes section
        summary = re.sub(r'^QUOTES:.*?\n\n', '', summary, flags=re.DOTALL)
    elif "סיכום:" in response:
        summary_part = response.split("סיכום:")[-1].strip()
        summary = summary_part.split("\n\n")[0].strip()
    else:
        # Fallback: take the last substantial paragraph
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip() and '"' not in p]
        if paragraphs:
            summary = paragraphs[-1]

    # Format final output
    result_parts = []
    for quote in quotes:
        # Clean quote
        quote = quote.strip()
        if len(quote) > 10:  # Only include substantial quotes
            result_parts.append(f'"{quote}"')

    if summary and len(summary) > 20:
        result_parts.append(f"\n{summary}")

    return "\n".join(result_parts) if result_parts else response[:300]


def extract_topics(model, tokenizer, description: str) -> List[str]:
    """Extract course topics from description"""

    if pd.isna(description) or not description.strip():
        return []

    prompt = TOPIC_EXTRACTION_PROMPT.format(description=description[:1000])

    print("\n" + "=" * 80)
    print("PROCESSING TOPIC EXTRACTION")
    print("=" * 80)

    try:
        response = generate_response(model, tokenizer, prompt, max_tokens=200)

        print(f"LLM Output:\n{response}")
        print("=" * 80)

        # Parse JSON response
        topics = parse_topics_response(response)
        return topics
    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def parse_topics_response(response: str) -> List[str]:
    """Parse the LLM response to extract topics list"""

    try:
        # Try to find JSON in the response
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            data = json.loads(json_match.group())
            if "topics" in data:
                topics = data["topics"]
                # Clean and validate
                topics = [t.strip() for t in topics if isinstance(t, str) and len(t.strip()) > 2]
                return topics[:6]

        # Fallback 1: Look for array format ["item", "item"]
        array_match = re.search(r'\[(.*?)\]', response)
        if array_match:
            items_str = array_match.group(1)
            items = re.findall(r'"([^"]+)"', items_str)
            if items:
                return [t.strip() for t in items if len(t.strip()) > 2][:6]

        # Fallback 2: Extract comma-separated items
        topics = []
        for line in response.split("\n"):
            if ":" in line:
                line = line.split(":", 1)[1]

            # Remove brackets and quotes
            line = re.sub(r'[\[\]{}"]', '', line)

            # Split by commas
            items = [item.strip() for item in line.split(",") if item.strip()]
            topics.extend(items)

        # Filter and deduplicate
        topics = [t for t in topics if 2 < len(t) < 50]
        seen = set()
        unique_topics = []
        for t in topics:
            t_lower = t.lower()
            if t_lower not in seen:
                seen.add(t_lower)
                unique_topics.append(t)

        return unique_topics[:6]

    except Exception as e:
        print(f"⚠ Parsing error: {e}")
        return []


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_course_data(df: pd.DataFrame, model, tokenizer, max_courses: int = None) -> pd.DataFrame:
    """Process all courses in the dataframe"""

    # Limit number of courses if specified
    if max_courses:
        df = df.head(max_courses)

    print(f"\nProcessing {len(df)} courses...")

    # Initialize new columns
    df['Review_summary'] = ""
    df['Course_Topic'] = None

    for idx, row in df.iterrows():
        print(f"\n{'=' * 80}")
        print(f"Processing Course {idx + 1}/{len(df)}")
        print(f"Title: {row.get('title', 'Unknown')}")
        print(f"Course ID: {row.get('course_id', 'N/A')}")
        print(f"{'=' * 80}")

        # Extract review summary
        if 'all_reviews' in df.columns and pd.notna(row['all_reviews']) and str(row['all_reviews']).strip():
            try:
                summary = extract_review_summary(model, tokenizer, row['all_reviews'])
                df.at[idx, 'Review_summary'] = summary
                print(f"✓ Review summary extracted ({len(summary)} chars)")
            except Exception as e:
                print(f"❌ Error processing reviews: {e}")
                df.at[idx, 'Review_summary'] = ""
        else:
            print("⊘ No reviews available")
            df.at[idx, 'Review_summary'] = ""

        # Extract topics
        if 'description' in df.columns and pd.notna(row['description']) and str(row['description']).strip():
            try:
                topics = extract_topics(model, tokenizer, row['description'])
                df.at[idx, 'Course_Topic'] = topics
                print(f"✓ Topics extracted: {topics}")
            except Exception as e:
                print(f"❌ Error extracting topics: {e}")
                df.at[idx, 'Course_Topic'] = []
        else:
            print("⊘ No description available")
            df.at[idx, 'Course_Topic'] = []

        # Save progress periodically
        if (idx + 1) % 5 == 0:
            df.to_csv('courses_data_progress.csv', index=False, encoding='utf-8')
            print(f"\n💾 Progress saved after {idx + 1} courses")

    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(test_mode: bool = False):
    """Main execution function"""

    print("=" * 80)
    print("COURSE REVIEW ANALYSIS PIPELINE")
    print("=" * 80)

    # Load data
    print("\nLoading CSV file...")
    df = pd.read_csv('courses_data_before_llm.csv', encoding='utf-8')
    print(f"✓ Loaded {len(df)} courses")
    print(f"  Columns: {list(df.columns)}")

    # Validate required columns
    required_cols = ['course_id', 'title']
    optional_cols = ['description', 'all_reviews']

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        return

    for col in optional_cols:
        if col in df.columns:
            non_empty = df[col].notna().sum()
            print(f"  {col}: {non_empty}/{len(df)} courses have data")

    # Load model
    model, tokenizer = load_model()

    # Process data (test mode: only 3 courses)
    max_courses = 3 if test_mode else None
    if test_mode:
        print(f"\n⚠ TEST MODE: Processing only {max_courses} courses")

    df_processed = process_course_data(df, model, tokenizer, max_courses)

    # Save results
    output_file = 'courses_data_processed.csv'
    df_processed.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n{'=' * 80}")
    print(f"✓ Processing complete! Results saved to: {output_file}")
    print(f"{'=' * 80}")

    # Display sample results
    print("\n📊 Sample Results:")
    print("-" * 80)
    for idx in range(min(3, len(df_processed))):
        row = df_processed.iloc[idx]
        print(f"\n{idx + 1}. {row.get('title', 'Unknown')}")
        print(f"   Topics: {row['Course_Topic']}")
        summary = row['Review_summary']
        if summary:
            print(f"   Summary preview: {summary[:150]}...")
        else:
            print(f"   Summary: (empty)")

    return df_processed


# Run in test mode first (set to False for full run)
if __name__ == "__main__":
    # Set to True to test on 3 courses first
    TEST_MODE = True

    result_df = main(test_mode=TEST_MODE)

    if TEST_MODE:
        print("\n" + "=" * 80)
        print("✓ Test completed successfully!")
        print("  To process all courses, set TEST_MODE = False")
        print("=" * 80)