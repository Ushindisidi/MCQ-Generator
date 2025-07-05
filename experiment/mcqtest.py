import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback # For token tracking

# NEW IMPORTS for LCEL (LangChain Expression Language)
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
# This serves as a strict guide for the LLM's output format.
RESPONSE_JSON = {
    "1": {
        "mcq": "What is the primary purpose of a virtual environment in Python?",
        "options": {
            "a": "To speed up code execution",
            "b": "To isolate project dependencies",
            "c": "To manage system-wide Python installations",
            "d": "To debug Python scripts more easily",
        },
        "correct": "b",
    },
    "2": {
        "mcq": "Which Git command is used to initialize a new local repository?",
        "options": {
            "a": "git start",
            "b": "git new",
            "c": "git init",
            "d": "git create",
        },
        "correct": "c",
    },
    "3": {
        "mcq": "LangChain helps in orchestrating what type of models?",
        "options": {
            "a": "Traditional machine learning models",
            "b": "Large Language Models (LLMs)",
            "c": "Image recognition models",
            "d": "Database management systems",
        },
        "correct": "b",
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

#  Define Individual Runnables using LCEL 
# These are the atomic operations.
quiz_chain_runnable = quiz_generation_prompt | llm | StrOutputParser()
review_chain_runnable = quiz_evaluation_prompt | llm | StrOutputParser()


#  Combine into a final chain using LCEL (REPLACES SequentialChain)
# I am using RunnablePassthrough and .assign() to manage inputs and outputs sequentially.

generate_evaluate_chain = (
    RunnableMap({
    # This dictionary ensures all expected variables are available for subsequent steps.
    
        "text": RunnablePassthrough(),
        "number": RunnablePassthrough(),
        "subject": RunnablePassthrough(),
        "tone": RunnablePassthrough(),
        "response_json": RunnablePassthrough(),
    })
    # Generate the quiz. The output of quiz_chain_runnable (the quiz string)
    # will be stored under the key "quiz_output" by .assign().
    # The original inputs are still part of the chain's state.
    .assign(quiz_output=quiz_chain_runnable)
    #  prepare inputs for the review_chain_runnable.
    .assign(review_output=
        RunnableParallel(
            subject=lambda x: x["subject"],    # Access 'subject' from the overall chain state (x)
            quiz=lambda x: x["quiz_output"],   # Access the 'quiz_output' from the previous step's result (x)
        )
        | review_chain_runnable # Pipe these prepared inputs to the review runnable
    )
    # This will be the final dictionary returned by generate_evaluate_chain.invoke().
    .pick(["quiz_output", "review_output"]))

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

    #  Convert RESPONSE_JSON to string for the prompt 
    json_response_str = json.dumps(RESPONSE_JSON)

    # Preparing input data for the LCEL chain 
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

    #  Running the LCEL Chain and Track Token Usage 
    with get_openai_callback() as cb:
        try:
            # Use .invoke() for LCEL chains to get the result
            response = generate_evaluate_chain.invoke(input_data)

            print(f"\n--- Token Usage & Cost ---")
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost:.6f}") # Format to 6 decimal places

            # Process and Print Outputs
            # The output of generate_evaluate_chain.invoke() is now a dictionary
            # with 'quiz_output' and 'review_output' keys (as defined by .pick())
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
                quiz_df = pd.DataFrame(parsed_quiz).T # Transpose to make question numbers as rows
                print(quiz_df)

                with open("Response.json", "w", encoding="utf-8") as f:
                    json.dump(parsed_quiz, f, indent=4)
                print("\nParsed quiz saved to Response.json")

            except json.JSONDecodeError as e:
                print(f"\nERROR: Failed to parse the quiz JSON output from the LLM.")
                print(f"JSON Decode Error: {e}")
                print(f"Check the raw output above for formatting issues. LLM might not have adhered to JSON strictly.")
            except Exception as e:
                print(f"\nAn unexpected error occurred while processing the quiz output: {e}")
                print(traceback.format_exc()) # Print full traceback for other errors

        except Exception as e:
            print(f"\nAn error occurred during the chain execution: {e}")
            print(traceback.format_exc()) # Print full traceback for errors during LLM call

import pandas as pd

# populating quiz_table_data and save to CSV 

parsed_quiz = {
    "1": {
        "mcq": "What is the primary function of a neuron?",
        "options": {"a": "To digest food", "b": "To transmit nerve impulses", "c": "To filter blood", "d": "To produce hormones"},
        "correct": "b"
    },
    "2": {
        "mcq": "Which part of the cell is responsible for generating energy through cellular respiration?",
        "options": {"a": "Nucleus", "b": "Ribosome", "c": "Mitochondria", "d": "Cell wall"},
        "correct": "c"
    }
}

quiz_table_data = []
if parsed_quiz: # Ensure parsed_quiz is not empty or None
    for key, value in parsed_quiz.items():
        # Defensive check for expected keys to avoid errors if LLM output is malformed
        if all(k in value for k in ["mcq", "options", "correct"]):
            mcq = value["mcq"]
            # Format options into a single string "a: Opt A | b: Opt B | c: Opt C | d: Opt D"
            options = " | ".join(
                [
                    f"{option_key}: {option_value}"
                    for option_key, option_value in value["options"].items()
                ]
            )
            correct = value["correct"]
            quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})
        else:
            print(f"Warning: Skipping malformed quiz question (missing 'mcq', 'options', or 'correct' keys): {value}")
else:
    print("No parsed quiz data available for table formatting.")

# Convert to DataFrame and save to CSV
if quiz_table_data: 
    quiz_df_formatted = pd.DataFrame(quiz_table_data)
    quiz_df_formatted.to_csv("machinelearning.csv", index=False) # Save to CSV
    print("\nFormatted quiz successfully saved to machinelearning.csv")

    #  Print the formatted DataFrame to console
    print("\n--- Formatted Quiz (for CSV) ---")
    print(quiz_df_formatted)
else:
    print("\nNo formatted quiz data available to save to CSV.")
