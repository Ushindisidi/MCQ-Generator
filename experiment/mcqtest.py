import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback# For token tracking

# IMPORTS for LCEL (LangChain Expression Language)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableMap
# from pypdf import PdfReader


#  Load Environment Variables 
load_dotenv() # take environment variables from .env.
KEY = os.getenv("OPENAI_API_KEY")

if not KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file. Please set it correctly.")

#  Initialize LLM
llm = ChatOpenAI(openai_api_key=KEY, model="gpt-3.5-turbo", temperature=0.5)
#  Define RESPONSE_JSON
RESPONSE_JSON = {
    "1": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "2": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "3": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
}
#  Define TEMPLATE 
TEMPLATE="""
You are a highly proficient and experienced educational expert specializing in creating challenging, clear, and well-structured Multiple Choice Questions (MCQs).
Your primary goal is to generate a quiz based on the provided TEXT for students studying a specific SUBJECT.

Given Information:
- TEXT: {text}
- NUMBER_OF_QUESTIONS: {number}
- SUBJECT_AREA: {subject}
- TONE_OF_QUESTIONS: {tone} (e.g., academic, conversational, formal, informal)

Your Instructions for MCQ Generation:
1.  Strictly generate {number} MCQs that are derived directly from the provided TEXT. If the text is insufficient, you may use general knowledge related to the SUBJECT_AREA to complete the required number of questions.
2.  Each MCQ must have exactly four (4) options, clearly labeled 'a', 'b', 'c', and 'd'.
3.  Ensure that only one option is the correct answer for each question.
4.  The incorrect options (distractors) should be plausible, relevant, and distinct from the correct answer to effectively test understanding.
5.  Questions must be concise, unambiguous, and cover important concepts or facts from the TEXT.
6.  Maintain the specified TONE_OF_QUESTIONS consistently throughout all generated questions and options.
7.  Do not repeat any questions or their options. Each question should be unique.

Output Format (CRITICAL - Adhere Strictly):
Provide your output as a JSON object that strictly adheres to the following RESPONSE_JSON format. *Do not include any preamble, postamble, explanations, conversational text, or markdown outside of the JSON block itself.* The response should be only the JSON.

### RESPONSE_JSON Format (Example and Guide):
{response_json}

Ensure the final output is a valid JSON object matching the RESPONSE_JSON structure.
"""

#  Define TEMPLATE(Quiz Evaluation Prompt) 
TEMPLATE2="""
You are an expert English grammarian and writer. Given a Multiple Choice Quiz for {subject} students.
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis.
If the quiz is not at par with the cognitive and analytical abilities of the students,
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities.
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""
#  Create PromptTemplates 
quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=TEMPLATE
)

quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"], 
    template=TEMPLATE2
)

# Define Individual Runnables using LCEL 
quiz_chain_runnable = quiz_generation_prompt | llm | StrOutputParser()
review_chain_runnable = quiz_evaluation_prompt | llm | StrOutputParser()

#  Combine into a final chain using LCEL
generate_evaluate_chain = (
    #  ensurimg all inputs are properly formatted for the quiz generation
    RunnablePassthrough()
    # Generate the quiz 
    .assign(quiz_output=quiz_chain_runnable)
    # Preparing inputs for the review chain
    .assign(review_output=
        RunnableMap({
            "subject": lambda x: x["subject"],
            "quiz": lambda x: x["quiz_output"], 
        })
        | review_chain_runnable
    )
    # Return both outputs
    .pick(["quiz_output", "review_output"])
)


def generate_and_evaluate_quiz(inputs):
    """
    Generate quiz and evaluation in a simple, reliable way
    """
    # Generate quiz
    quiz_result = quiz_chain_runnable.invoke(inputs)
    
    # Preparing review inputs
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

generate_evaluate_chain = (
    {
        "text": lambda x: x["text"],
        "number": lambda x: x["number"], 
        "subject": lambda x: x["subject"],
        "tone": lambda x: x["tone"],
        "response_json": lambda x: x["response_json"],
    }
    | RunnablePassthrough.assign(
        quiz_output=quiz_chain_runnable
    )
    | RunnablePassthrough.assign(
        review_output=RunnableMap({
            "subject": lambda x: x["subject"],
            "quiz": lambda x: x["quiz_output"],
        }) | review_chain_runnable
    )
    | RunnableMap({
        "quiz_output": lambda x: x["quiz_output"],
        "review_output": lambda x: x["review_output"],
    })
)

#  Main Execution Block
if __name__ == "__main__":
    # --- Load Text from data.txt ---
    file_path = r"C:\Users\Admin\Desktop\MCQ-Generator\data.txt"
    TEXT = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            TEXT = file.read()
        print(f"Successfully loaded text from {file_path}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it exists in the project root.")
        exit()
    except Exception as e:
        print(f"An error occurred while reading '{file_path}': {e}")
        exit()

    # --- Define Quiz Parameters ---
    NUMBER = 1     
    SUBJECT = "biology"
    TONE = "simple"

    # Convert RESPONSE_JSON to string for the prompt 
    json_response_str = json.dumps(RESPONSE_JSON)

    # Preparing input data for the chain 
    input_data = {
        "text": TEXT,
        "number": NUMBER,
        "subject": SUBJECT,
        "tone": TONE,
        "response_json": json_response_str
    }

    print(f"\n--- Starting MCQ Generation and Tracking Token Usage ---")
    print(f"Text length: {len(TEXT)} characters")
    print(f"Requested: {NUMBER} MCQs for {SUBJECT} students in a {TONE} tone.")

    # Running the Chain and Track Token Usage 
    with get_openai_callback() as cb:
        try:
            response = generate_and_evaluate_quiz(input_data)

            print(f"\n--- Token Usage & Cost ---")
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost:.6f}")

            # Process and Print Outputs
            quiz_output_raw = response.get("quiz_output")
            review_output = response.get("review_output")

            print("\n--- Generated Quiz (Raw LLM Output) ---")
            print(quiz_output_raw)

            print("\n--- Evaluation Review ---")
            print(review_output)

            # Attempt to parse the quiz JSON output from the LLM
            try:
                parsed_quiz = json.loads(quiz_output_raw)
                print("\n--- Parsed Quiz (DataFrame) ---")
                
                # Convert parsed_quiz dict to a DataFrame for display
                quiz_table_data = []
                
                if isinstance(parsed_quiz, dict):
                    for key, value in parsed_quiz.items():
                        if isinstance(value, dict) and all(k in value for k in ["mcq", "options", "correct"]):
                            mcq = value["mcq"]
                            
                            if isinstance(value["options"], dict):
                                options = " | ".join(
                                    [f"{option_key}: {option_value}" for option_key, option_value in value["options"].items()]
                                )
                            else:
                                options = str(value["options"])
                            
                            correct = value["correct"]
                            quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})
                        else:
                            print(f"Warning: Skipping malformed quiz question for key '{key}': {value}")
                
                if quiz_table_data:
                    quiz_df_formatted = pd.DataFrame(quiz_table_data)
                    print(quiz_df_formatted)

                    with open("Response.json", "w", encoding="utf-8") as f:
                        json.dump(parsed_quiz, f, indent=4)
                    print("\nParsed quiz saved to Response.json")
                    
                    # Save to CSV
                    quiz_df_formatted.to_csv("Biology.csv", index=False)
                    print("Formatted quiz successfully saved to Biology.csv")
                else:
                    print("No valid quiz data found to display.")

            except json.JSONDecodeError as e:
                print(f"\nERROR: Failed to parse the quiz JSON output from the LLM.")
                print(f"JSON Decode Error: {e}")
                print(f"Raw output that failed to parse:")
                print(f"'{quiz_output_raw}'")
                print(f"Check the raw output above for formatting issues.")
            except Exception as e:
                print(f"\nAn unexpected error occurred while processing the quiz output: {e}")
                print(f"Raw quiz output: {quiz_output_raw}")
                print(traceback.format_exc())

        except Exception as e:
            print(f"\nAn error occurred during the chain execution: {e}")
            print(traceback.format_exc())