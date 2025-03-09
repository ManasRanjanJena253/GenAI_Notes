# Importing dependencies
import os
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator

# Loading environment variables from .env file
load_dotenv()

# Defining data model for MCQs using pydantic module.
class MCQs(BaseModel):
    # Defining the structure of an mcq question using field
    question: str = Field(description = "The question text.")
    options : List[str] = Field(description = 'List of 4 possible answers.')
    correct_answer : str = Field(description = 'The correct answer of the question.')

    # Custom validator to convert the question into pure string type if by chance it is in any other format such as dict or int.
    @field_validator('question')
    def clean_question(cls, v):
        if isinstance(v, dict):
            return v.get('description', str(v))   # The result if in the form of dict it will be as {'dictionary' : "Question statement"}.
        return str(v)

# Defining a data model for fill in the blank questions
class FillInTheBlanks(BaseModel):
    # Defining the structure of fill in the blank questions.
    question : str = Field(description = "The question text with '_________' for the blank.")
    answer : str = Field(description = "The correct word or phrase for the blank.")

    @field_validator('question')
    def clean_question(cls, v):
        if isinstance(v, dict):
            return v.get('description', str(v))
        return str(v)

class QuestionGenerator:
    def __init__(self):
        """
        Initialize question generator with Groq API
        Sets up the llm with specific parameters
        """
        self.llm = ChatGroq(
            api_key = os.getenv('GROQ_API_KEY'),
            model = "llama-3.1-8b-instant",
            temperature = 0.8
        )

# Defining multiple helper functions
    def generate_mcq(self, topic: str, difficulty : str = 'Medium') -> MCQs: # ->MCQs means that this function will return its values to the MCQs class that we defined.
        """Function to generate multiple choice questions. """
        # Setting up Pydantic parser for type checking and validation
        mcq_parser = PydanticOutputParser(pydantic_object = MCQs)

        # Defining the prompt template with specific format requirements
        prompt = PromptTemplate(
            template = ("Generate a {difficulty} multiple choice question about {topic}."
                        "Return ONLY a JSON Object with these exact fields:\n"
                        "'question' : A clear, specific question\n"
                        "'options' : An array of exactly 4 possible answers\n"
                        "'correct_answer' : One of the options that is correct with its complete reasoning and explanation on why it is the correct answer."
                        "Example format : \n"
                        "{{\n"
                        "'question' : 'What is the capital of france?',\n "
                        "'options' : ['London','America', 'Australia', 'Paris'],\n"
                        "'correct_answer' : 'Paris'\n"
                        "}}\n"
                        "Your Response : "
        ), input_variables = ['topic', 'difficulty'])

        # Implementing retry method for llm to know how much time it should try to generate the questions.
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Generating a response using LLM
                response = self.llm.invoke(prompt.format(topic = topic, difficulty = difficulty))
                parsed_response = mcq_parser.parse(response.content)

                # Checking if the generated question meets the requirements.
                if not parsed_response.question or len(parsed_response.options) != 4 or not parsed_response.correct_answer:
                    raise ValueError("Invalid Question Format")
                if parsed_response.correct_answer not in parsed_response.options:
                    raise ValueError("Correct Answer not in options")
                return parsed_response

            except Exception as e:
                # On final attempt, raise error, otherwise continue trying
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to generate valid MCQ after {max_attempts} attempts : {str(e)}")
                continue

    def generate_fill_blank(self, topic : str, difficulty : str = 'Medium'):
        """Function to generate multiple choice questions. """
        # Setting up Pydantic parser for type checking and validation
        fill_blank_parser = PydanticOutputParser(pydantic_object = FillInTheBlanks)

        prompt = PromptTemplate(
            template = ("Generate a {difficulty} fill-in-the-blank question about {topic}.\n\n"
            "Return ONLY a JSON object with these exact fields : \n"
            "'question' : A sentence with '________' marking where the blank should be\n"
            "'answer' : The correct word o phrase that belongs in the blank\n\n"
            "Example format:\n"
            "{{\n"
            "'question' : 'The capital of france is ________ ',\n"
            "'answer' : 'Paris'\n"
            "}}\n"
            "Your Response : "
        ),input_variables = ['topic', 'difficulty'])

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Generating a response using LLM
                response = self.llm.invoke(prompt.format(topic = topic, difficulty = difficulty))
                parsed_response = fill_blank_parser.parse(response.content)

                # Checking if the generated question meets the requirements.
                if not parsed_response.question or not parsed_response.answer:
                    raise ValueError("Invalid Question Format")
                if '________' not in parsed_response.answer:
                    raise ValueError("Correct Answer not in options")
                return parsed_response

            except Exception as e:
                # On final attempt, raise error, otherwise continue trying
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to generate valid Fill in the blanks paper after {max_attempts} attempts : {str(e)}")
                continue



