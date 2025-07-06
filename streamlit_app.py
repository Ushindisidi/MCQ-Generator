import streamlit as st
import os
import json
import pandas as pd
from dotenv import load_dotenv
import traceback
import re

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_core.output_parsers import StrOutputParser

# --- Configuration & Initialization ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in .env file. Please set it correctly.")
    st.stop()

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0.2)

# Define RESPONSE_JSON template
RESPONSE_JSON = {
    "1": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here", 
            "c": "choice here",
            "d": "choice here"
        },
        "correct": "correct answer"
    },
    "2": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here", 
            "d": "choice here"
        },
        "correct": "correct answer"
    },
    "3": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here"
        },
        "correct": "correct answer"
    }
}

# Define PromptTemplates
quiz_generation_prompt = PromptTemplate(
    template="""You are a highly proficient and experienced educational expert specializing in creating challenging, clear, and well-structured Multiple Choice Questions (MCQs).
Your primary goal is to generate a quiz based on the provided TEXT for students studying a specific SUBJECT.

Given Information:
- TEXT: {text}
- NUMBER_OF_QUESTIONS: {number}
- SUBJECT_AREA: {subject}
- TONE_OF_QUESTIONS: {tone} (e.g., academic, conversational, formal, informal)

Your Instructions for MCQ Generation:
1. Strictly generate {number} MCQs that are derived directly from the provided TEXT. If the text is insufficient, you may use general knowledge related to the SUBJECT_AREA to complete the required number of questions.
2. Each MCQ must have exactly four (4) options, clearly labeled 'a', 'b', 'c', and 'd'.
3. Ensure that only one option is the correct answer for each question.
4. The incorrect options (distractors) should be plausible, relevant, and distinct from the correct answer to effectively test understanding.
5. Questions must be concise, unambiguous, and cover important concepts or facts from the TEXT.
6. Maintain the specified TONE_OF_QUESTIONS consistently throughout all generated questions and options.
7. Do not repeat any questions or their options. Each question should be unique.

Output Format (CRITICAL - Adhere Strictly):
You must return ONLY a valid JSON object. Do not include any explanatory text, markdown formatting, or code blocks.
Your response must start with {{ and end with }}. Nothing else.

The JSON must follow this exact structure:
{response_json}

IMPORTANT: 
- Use double quotes for all strings
- Do not use trailing commas
- Ensure proper JSON syntax
- Do not include any text before or after the JSON object
- Each question should be numbered as a string ("1", "2", "3", etc.)
""",
    input_variables=["text", "number", "subject", "tone", "response_json"]
)

quiz_evaluation_prompt = PromptTemplate(
    template="""You are an expert English grammarian and writer. Given a Multiple Choice Quiz for {subject} students.
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis.
If the quiz is not at par with the cognitive and analytical abilities of the students,
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities.

Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
""",
    input_variables=["subject", "quiz"]
)

quiz_chain_runnable = quiz_generation_prompt | llm | StrOutputParser()
review_chain_runnable = quiz_evaluation_prompt | llm | StrOutputParser()

def generate_and_evaluate_quiz(inputs):
    """Generate quiz and evaluation in a simple, reliable way"""
    # Generate quiz
    quiz_result = quiz_chain_runnable.invoke(inputs)
    
    # Prepare review inputs
    review_inputs = {
        "subject": inputs["subject"],
        "quiz": quiz_result
    }
    
    # Generate review
    review_result = review_chain_runnable.invoke(review_inputs)
    
    return {
        "quiz_output": quiz_result,
        "review_output": review_result
    }

def clean_and_parse_json(json_string):
    """Clean and parse JSON string from LLM response with multiple fallback strategies"""
    try:
        # Remove any markdown formatting
        json_string = json_string.strip()
        
        # Remove common markdown patterns
        if json_string.startswith("```json"):
            json_string = json_string[7:]
        elif json_string.startswith("```"):
            json_string = json_string[3:]
        
        if json_string.endswith("```"):
            json_string = json_string[:-3]
        
        # Remove any leading/trailing whitespace
        json_string = json_string.strip()
        
        # Find JSON boundaries if there's extra text
        start_idx = json_string.find('{')
        end_idx = json_string.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_string = json_string[start_idx:end_idx]
        
        # Fix common JSON issues
        json_string = fix_common_json_issues(json_string)
        
        # Parse JSON
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {e}")
        st.error(f"Problematic JSON string: {json_string[:500]}...")
        return None
    except Exception as e:
        st.error(f"Unexpected error while parsing JSON: {e}")
        return None

def fix_common_json_issues(json_str):
    """Fix common JSON formatting issues"""
    # Remove trailing commas before closing braces
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Fix single quotes to double quotes (basic fix)
    json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
    json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
    
    return json_str

def validate_quiz_structure(quiz_data):
    """Validate that the quiz has the expected structure"""
    if not isinstance(quiz_data, dict):
        return False, "Quiz data is not a dictionary"
    
    required_keys = ["mcq", "options", "correct"]
    
    for q_num, question in quiz_data.items():
        if not isinstance(question, dict):
            return False, f"Question {q_num} is not a dictionary"
        
        for key in required_keys:
            if key not in question:
                return False, f"Question {q_num} missing '{key}' field"
        
        if not isinstance(question["options"], dict):
            return False, f"Question {q_num} options is not a dictionary"
        
        expected_options = {"a", "b", "c", "d"}
        actual_options = set(question["options"].keys())
        if actual_options != expected_options:
            return False, f"Question {q_num} has incorrect option keys: {actual_options}"
    
    return True, "Valid structure"

# --- Streamlit UI Components ---
st.set_page_config(
    page_title="MCQ Generator",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🧠 AI-Powered MCQ Generator")
st.markdown("Generate multiple-choice questions from your text using advanced AI.")

# Helpful information
with st.expander("ℹnfo:How to use this tool"):
    st.markdown("""
    1. **Enter your text**: Paste the content you want to create questions from
    2. **Set parameters**: Choose number of questions, subject, and tone
    3. **Generate**: Click the button and wait for AI to create your quiz
    4. **Download**: Get your quiz as a CSV file for use in other applications
    """)

# Input components
input_text = st.text_area(
    "Paste your text here (e.g., lecture notes, article excerpts):", 
    height=250, 
    help="The source material for your MCQs. Longer texts generally produce better questions.",
    placeholder="Enter your educational content here..."
)

col1, col2 = st.columns(2)
with col1:
    num_questions = st.slider(
        "Number of MCQs to generate:", 
        min_value=1, 
        max_value=10, 
        value=3,
        help="More questions take longer to generate"
    )
    subject = st.text_input(
        "Subject of the questions:", 
        value="General Knowledge",
        help="e.g., Biology, Machine Learning, History"
    )

with col2:
    tone = st.selectbox(
        "Tone of the questions:", 
        ["simple", "academic", "conversational", "formal"],
        help="Simple: Easy to understand, Academic: Formal and precise"
    )

# Generate button
if st.button(" Generate MCQs", type="primary"):
    if not input_text.strip():
        st.warning("⚠️ Please provide some text to generate MCQs.")
        st.stop()

    if len(input_text.strip()) < 5:
        st.warning("⚠️ Your text seems quite short. Consider providing more content for better questions.")

    with st.spinner("🤖 Generating and evaluating MCQs... This may take a moment."):
        try:
            # Create a clean JSON string for the template
            json_response_str = json.dumps(RESPONSE_JSON, indent=2)
            
            input_data = {
                "text": input_text,
                "number": num_questions,
                "subject": subject,
                "tone": tone,
                "response_json": json_response_str
            }

            with get_openai_callback() as cb:
                response = generate_and_evaluate_quiz(input_data)
                
                # Display token usage in sidebar
                with st.sidebar:
                    st.subheader("📊 Token Usage & Cost")
                    st.metric("Total Tokens", cb.total_tokens)
                    st.metric("Prompt Tokens", cb.prompt_tokens)
                    st.metric("Completion Tokens", cb.completion_tokens)
                    st.metric("Total Cost (USD)", f"${cb.total_cost:.6f}")

            quiz_output_raw = response.get("quiz_output")
            review_output = response.get("review_output")

            # Debug: Show raw output
            with st.expander("🔍 Raw AI Output (for debugging)"):
                st.text_area("Raw AI Output:", quiz_output_raw, height=200)

            # Parse the quiz JSON
            parsed_quiz = clean_and_parse_json(quiz_output_raw)
            
            if parsed_quiz is None:
                st.error("❌ Failed to parse the quiz JSON. Please try again.")
                st.stop()

            # Validate quiz structure
            is_valid, validation_message = validate_quiz_structure(parsed_quiz)
            if not is_valid:
                st.error(f"❌ Invalid quiz structure: {validation_message}")
                st.stop()

            # Process the quiz data
            quiz_table_data = []
            if isinstance(parsed_quiz, dict):
                for key, value in parsed_quiz.items():
                    if isinstance(value, dict) and all(k in value for k in ["mcq", "options", "correct"]):
                        mcq = value["mcq"]
                        if isinstance(value["options"], dict):
                            options = " | ".join(
                                [f"{ok.upper()}: {ov}" for ok, ov in value["options"].items()]
                            )
                        else:
                            options = "Malformed Options"
                            st.warning(f"⚠️ Malformed options for question {key}")

                        correct = value["correct"]
                        quiz_table_data.append({
                            "Question": mcq, 
                            "Options": options, 
                            "Correct Answer": correct
                        })
                    else:
                        st.warning(f"⚠️ Skipping malformed question {key}")

            if quiz_table_data:
                st.success(f"✅ Successfully generated {len(quiz_table_data)} MCQs!")
                
                # Display the quiz
                st.subheader("📝 Generated Quiz")
                quiz_df_formatted = pd.DataFrame(quiz_table_data)
                st.dataframe(quiz_df_formatted, use_container_width=True)

                # Provide download link for CSV
                csv = quiz_df_formatted.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Quiz as CSV",
                    data=csv,
                    file_name=f"mcq_quiz_{subject.replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                )

                # Display evaluation
                st.subheader("📋 Evaluation Review")
                st.write(review_output)

                # Save quiz data 
                st.session_state.last_quiz = {
                    "quiz_data": quiz_table_data,
                    "subject": subject,
                    "review": review_output
                }

            else:
                st.error("❌ No valid quiz questions were generated. Please try again with different text or parameters.")

        except Exception as e:
            st.error(f"❌ An error occurred during quiz generation: {str(e)}")
            with st.expander("🔍 Error Details"):
                st.text(traceback.format_exc())

# Display previous quiz if available
if "last_quiz" in st.session_state:
    with st.expander("📚 Previous Quiz"):
        last_quiz_df = pd.DataFrame(st.session_state.last_quiz["quiz_data"])
        st.dataframe(last_quiz_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and OpenAI | [Source Code](https://github.com/Ushindisidi)")