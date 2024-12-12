from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from langchain_groq import ChatGroq
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.llms import ChatMessage
from langchain_huggingface import HuggingFaceEmbeddings


embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


groq_api_key = userdata.get('GROQ_API_KEY')

docs_both = SimpleDirectoryReader(input_files=["/content/data/PythonTB.pdf"]).load_data()

#BAAI/bge-small-en-v1.5
#sentence-transformers/all-mpnet-base-v2
#sentence-transformers/all-MiniLM-L6-V2
#embed_model =  HuggingFaceEmbeddings(model_name = "BAAI/bge-small-en-v1.5")

from llama_index.llms.groq import Groq

#llm = Groq(model="llama3-8b-8192") #  llama-3.1-8b-instant

llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name='llama3-8b-8192'
)
llm_70b = Groq(model="llama3-70b-8192")

Settings.llm = llm
Settings.embed_model = embed_model

messages = [
    ChatMessage(role="system", content="You are Athena, a friendly and supportive computer science teacher specializing in algorithms, data structures, and problem-solving, representing wisdom. Answer only computer science questions, avoid unrelated topics, and do not provide full answers to problem sets or coding solutions to maintain academic honesty. Instead, offer comments on how students can proceed or improve using bullet points, theoretical examples, and highlighting keywords in **bold**. If errors are spotted, mention the line numbers. Use simple language suitable for first-time learners, provide concise answers, avoid giving direct coding answers or corrected code, and allow follow-up questions without citing sources."),
    ChatMessage(role="user", content="Ask computer science related questions"),
]

index = VectorStoreIndex.from_documents(docs_both)
query_engine = index.as_query_engine(similarity_top_k=3)

response = query_engine.query("What is a string?")
print(response)

from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFDirectoryLoader
from google.colab import userdata
import os
import time
import textwrap
import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings

loader = PyPDFDirectoryLoader("data")
the_text = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(the_text)

vectorstore = Chroma.from_documents(
    documents=chunks,
    collection_name="rag_embeds",
   embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-V2"),
)
retriever = vectorstore.as_retriever()

groq_api_key = userdata.get('GROQ_API_KEY')

llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name='llama3-8b-8192'
    )


rag_template = """You are a friendly and supportive computer science teacher that experts in teaching algorithms, data structures, and problem-solving.
Your name is Athena and you represent wisdom. You don't know how to code: You only know how to explain.
Do not answer questions about topics other than computer science.
Do not provide full answers to problem sets as this would violate academic honesty.
Do not give coding solutions to the problems.
Use bullet points to point out the steps. Give theoretical examples of the techniques used but don't give the solution.
Never include coding examples in the response.
Provide a summary in the beginning and make the keywords in **bold** using markdown format.
If errors are spotted, give line numbers of the errors too.
Use language that could be easily understood by first-time programming learners. Never give direct answers to coding questions.
Don't give any corrected versions of the wrong code provided by the user.
Keep your answers short to the point. Answer only the main idea of the question.
Allow the user to ask follow-up questions.
Don't state where you extracted your answer from.
Answer the question based only on the following context:
{context}
Question: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)


response = rag_chain.invoke("What is this document about")
print(textwrap.fill(response, width=80))


# Make the questions dynamic using a chat interface. Let's use gradio for this.
def process_question(user_question):
    start_time = time.time()

    # Directly using the user's question as input for rag_chain.invoke
    response = rag_chain.invoke(user_question)

    # Measure the response time
    end_time = time.time()
    response_time = f"Response time: {end_time - start_time:.2f} seconds."

    # Combine the response and the response time into a single string
    full_response = f"{response}\n\n{response_time}"

    return full_response

# Setup the Gradio interface
iface = gr.Interface(fn=process_question,
                     inputs=gr.Textbox(lines=2, placeholder="Type your question here..."),
                     outputs=gr.Textbox(),
                     title="Athena - Your Educational AI Assistant",
                     description="Ask any question about your document, and get an answer along with the response time.")

# Launch the interface
iface.launch()

context_prompt = """
  The following is a friendly conversation between a user and an AI assistant named Athena.
  If the assistant does not know the answer to a question, it truthfully says it
  does not know. You don't give coding answers to the questions.

  Here are the relevant documents for the context:

  {context_str}

  Instruction: Based on the above documents, provide short-to-the-point explanations for the user question below but do not provide direct answers and coding solutions. Don't state where you extracted the answers from.
  Answer "don't know" if not present in the document.
  """
system_prompt = """You are a friendly and supportive computer science teacher that experts in teaching algorithms,
data structures, and problem-solving.
Your name is Athena and you represent wisdom. You don't know how to code: You only know how to explain.
Do not answer questions about topics other than computer science.
Do not provide full answers to problem sets as this would violate academic honesty.
Do not give coding solutions to the problems.
Use bullet points to point out the steps. Give theoretical examples of the techniques used but don't give the
solution.
Never include coding examples in the response.
Provide a summary in the beginning and make the keywords in bold.
If errors are spotted, give line numbers of the errors too.
Use language that could be easily understood by first-time programming learners. Never give direct answers to
coding questions.
Don't give any corrected versions of the wrong code provided by the user.
Keep your answers short to the point. Answer only the main idea of the question.
Allow the user to ask follow-up questions.
Don't state where you extracted your answer from.
Don't give coding examples, solutions or explanations.
You don't know how to code. You only know how to explain.
Don't apologize for not providing answers and coding solutions.
"""


memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    system_prompt=system_prompt,
    memory=memory,
    llm=llm,
    context_prompt= context_prompt,
    verbose=False,
)

print(chat_engine.chat("what is a string?"))

import base64
import io
import gradio as gr
from groq import Groq
from PIL import Image
import requests

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_image(image, prompt, is_url=False):
    client = Groq(api_key="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")

    if is_url:
        image_content = {"type": "image_url", "image_url": {"url": image}}
    else:
        base64_image = encode_image(image)
        image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        image_content,
                    ],
                }
            ],
            model="llava-v1.5-7b-4096-preview",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"



def process_image(image, url, prompt):
    if image is not None:
        return analyze_image(image, prompt)
    elif url:
        try:
            response = requests.get(url)
            image = Image.open(io.BytesIO(response.content))
            return analyze_image(url, prompt, is_url=True)
        except:
            return "Invalid image URL. Please provide a direct link to an image."
    else:
        return "Please provide an image to analyze."

# Function to update the prompt based on the selected topic
def update_prompt(topic):
    return f"Generate 3 revision coding questions on {topic}"

# Function to process the initial question in the "Ask a Question" tab
def process_question_with_image(image, question):
    if image:
         return analyze_image(image, question)
    else:
        response = chat_engine.chat(question)
        return str(response)

# Function to process the initial question in the "Revise" tab
def process_question_no_image(question):
    response = chat_engine.chat(question)
    return str(response)

# List of topics
beginner_topics = [
    "Variables and Data Types",
    "Control Structures",
    "Functions",
    "Basic Input/Output",
    "Lists and Arrays",
    "Dictionaries",
    "String Manipulation",
    "Basic Error Handling",
    "Basic File Operations",
    "Introduction to Object-Oriented Programming (OOP)",
    "Basic Algorithm Concepts",
    "Introduction to Libraries and Modules"
]

advanced_topics = [
    "Advanced Data Structures (Trees, Graphs)",
    "Algorithms and Complexity (Big O Notation)",
    "Recursion",
    "Object-Oriented Design Principles",
    "Design Patterns",
    "Functional Programming Concepts",
    "Concurrency and Multithreading",
    "Memory Management and Optimization",
    "Network Programming",
    "Database Design and SQL",
    "Web Development Frameworks",
    "Machine Learning Basics"
]

# Create the Gradio interfaces
with gr.Blocks() as iface1:
    with gr.Column():
        gr.Markdown("# Athena - AI Programming Assistant")
        gr.Markdown("## Ask any question related to computer science.")
        image_input = gr.Image(type="pil", label="Upload an Image (Provide a brief description for more precise responses!)", elem_id="image-input")
        input_box = gr.Textbox(lines=2, placeholder="Type your question here...", label="Question")
        output_box = gr.Markdown()
        generate_button = gr.Button("Generate")
        generate_button.click(fn=process_question_with_image, inputs=[image_input, input_box], outputs=output_box)

with gr.Blocks() as iface2:
    with gr.Column():
        gr.Markdown("Test your skills here by asking Athena to give you revision questions. You can choose your topic from the buttons below or ask about other topics.")
        input_box2 = gr.Textbox(lines=2, placeholder="Generate revision questions on the topic here...", label="Revision Question")
        output_box2 = gr.Markdown()
        generate_button2 = gr.Button("Generate")
        gr.Markdown("### Beginner Topics")
        with gr.Row():
            for topic in beginner_topics:
                button = gr.Button(topic, elem_id="topic-button")
                button.click(fn=update_prompt, inputs=gr.Textbox(value=topic, visible=False), outputs=input_box2)
        gr.Markdown("### Advanced Topics")
        with gr.Row():
            for topic in advanced_topics:
                button = gr.Button(topic, elem_id="topic-button")
                button.click(fn=update_prompt, inputs=gr.Textbox(value=topic, visible=False), outputs=input_box2)
        generate_button2.click(fn=process_question_no_image, inputs=[input_box2], outputs=output_box2)

# Combine all interfaces into a tabbed interface
tabbed_iface = gr.TabbedInterface(
    interface_list=[iface1, iface2],
    tab_names=["Ask a Question", "Revise"],
    theme=gr.Theme.from_hub('HaleyCH/HaleyCH_Theme')
)


tabbed_iface.launch(inline=False, inbrowser=True, show_error=True, debug=True)

#https://huggingface.co/spaces/Yadanar1010/athena-ai-programming-mentor
