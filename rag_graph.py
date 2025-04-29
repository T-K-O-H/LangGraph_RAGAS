from typing import Dict, List, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import Graph, StateGraph, END
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
from ragas import evaluate
from datasets import Dataset
import os
from dotenv import load_dotenv
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Define the state
class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "The messages in the conversation"]
    context: Annotated[str, "The retrieved context"]
    response: Annotated[str, "The generated response"]
    metrics: Annotated[Dict, "The RAGAS metrics"]
    next: str

# Initialize components
llm = ChatOpenAI(model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize Chroma with minimal configuration
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="rag_collection",
    collection_metadata={"hnsw:space": "cosine"}
)

# Define the retrieval function
def retrieve(state: AgentState) -> Dict:
    try:
        messages = state["messages"]
        last_message = messages[-1]
        logger.info(f"Retrieving context for query: {last_message.content}")
        
        # Get relevant documents
        docs = vectorstore.similarity_search_with_score(
            last_message.content,
            k=10  # Increased number of documents
        )
        if not docs:
            logger.warning("No relevant documents found in the knowledge base")
            raise ValueError("No relevant documents found in the knowledge base")

        # Filter and combine documents - using a lower threshold
        filtered_docs = []
        for doc, score in docs:
            if score > 0.2:  # Lower threshold for more context
                filtered_docs.append(doc)

        if not filtered_docs:
            logger.warning("No documents met the similarity threshold, using all retrieved documents")
            filtered_docs = [doc for doc, _ in docs]  # Use all documents if none meet threshold

        # Sort documents by relevance (using the original scores)
        sorted_docs = sorted(zip(filtered_docs, [score for _, score in docs if score > 0.2]), 
                            key=lambda x: x[1], reverse=True)
        context = "\n\n".join([doc.page_content for doc, _ in sorted_docs])
        logger.info(f"Using {len(filtered_docs)} documents for context")
        
        # Validate context
        if not context.strip():
            logger.warning("No valid context could be retrieved")
            raise ValueError("No valid context could be retrieved")
            
        logger.info(f"Retrieved context length: {len(context)} characters")
        return {
            "context": context,
            "metrics": {},  # Initialize empty metrics
            "next": "generate"
        }
    except Exception as e:
        logger.error(f"Error in retrieval: {str(e)}")
        return {
            "context": "",
            "metrics": {
                "error": str(e),
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_correctness": 0.0
            },
            "next": "generate"
        }

# Define the generation function
def generate(state: AgentState) -> Dict:
    try:
        messages = state["messages"]
        context = state["context"]
        
        if not context.strip():
            logger.warning("Empty context in generation step")
            return {
                "response": "I apologize, but I couldn't find any relevant information in the knowledge base to answer your question. Please try rephrasing your question or upload more relevant documents.",
                "metrics": {
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "context_precision": 0.0,
                    "context_recall": 0.0,
                    "answer_correctness": 0.0,
                    "note": "No context available for evaluation"
                },
                "next": "evaluate"
            }
        
        logger.info("Generating response with context")
        # Create prompt with context
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant specialized in quantum computing. 
            Use the following context to answer the question. 
            Guidelines:
            1. Base your answer strictly on the provided context
            2. If the context doesn't contain relevant information, say so clearly
            3. Be specific and technical in your explanations
            4. Use bullet points or numbered lists when appropriate
            5. Include relevant examples from the context
            6. If discussing technical concepts, explain them clearly
            
            Context:
            {context}"""),
            ("human", "{question}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "context": context,
            "question": messages[-1].content
        })
        
        logger.info(f"Generated response length: {len(response)} characters")
        
        # Calculate metrics directly in generate
        try:
            logger.info("Creating dataset for metrics calculation")
            dataset = Dataset.from_dict({
                "question": [messages[-1].content],
                "contexts": [[context]],
                "answer": [response],
                "ground_truth": [context]
            })
            
            logger.info("Calculating RAGAS metrics")
            metrics_dict = {}
            result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness])
            
            metrics_dict["faithfulness"] = float(np.mean(result["faithfulness"]))
            metrics_dict["answer_relevancy"] = float(np.mean(result["answer_relevancy"]))
            metrics_dict["context_precision"] = float(np.mean(result["context_precision"]))
            metrics_dict["context_recall"] = float(np.mean(result["context_recall"]))
            metrics_dict["answer_correctness"] = float(np.mean(result["answer_correctness"]))
            
            logger.info(f"RAGAS metrics calculated: {metrics_dict}")
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            metrics_dict = {
                "error": str(e),
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_correctness": 0.0
            }
        
        return {
            "response": response,
            "metrics": metrics_dict,
            "next": "evaluate"
        }
    except Exception as e:
        logger.error(f"Error in generation: {str(e)}")
        return {
            "response": "I apologize, but I encountered an error while generating a response. Please try again.",
            "metrics": {
                "error": str(e),
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_correctness": 0.0
            },
            "next": "evaluate"
        }

# Define the RAGAS evaluation function
def evaluate_rag(state: AgentState) -> Dict:
    try:
        messages = state["messages"]
        context = state["context"]
        response = state["response"]
        
        # Detailed logging of input data
        logger.info("=== RAGAS Evaluation Debug Info ===")
        logger.info(f"Question: {messages[-1].content}")
        logger.info(f"Context length: {len(context)}")
        logger.info(f"Response length: {len(response)}")
        logger.info(f"Context preview: {context[:200]}...")
        logger.info(f"Response preview: {response[:200]}...")
        
        # Check if metrics are already in state
        if "metrics" in state:
            logger.info(f"Metrics found in state: {state['metrics']}")
            return {"context": context, "response": response, "metrics": state["metrics"], "next": END}
        
        # Validate inputs
        if not context.strip():
            logger.error("Empty context detected")
            return {"context": context, "response": response, "metrics": {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_correctness": 0.0,
                "note": "Empty context"
            }, "next": END}
            
        if not response.strip():
            logger.error("Empty response detected")
            return {"context": context, "response": response, "metrics": {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_correctness": 0.0,
                "note": "Empty response"
            }, "next": END}
        
        logger.info("Creating evaluation dataset...")
        try:
            # Create dataset for evaluation
            dataset = Dataset.from_dict({
                "question": [messages[-1].content],
                "contexts": [[context]],
                "answer": [response],
                "ground_truth": [context]  # Use context as ground truth for better evaluation
            })
            logger.info("Dataset created successfully")
            
            # Initialize metrics dictionary
            metrics_dict = {}
            
            # Evaluate each metric separately
            try:
                result = evaluate(dataset, metrics=[faithfulness])
                metrics_dict["faithfulness"] = float(np.mean(result["faithfulness"]))
                logger.info(f"Faithfulness calculated: {metrics_dict['faithfulness']}")
            except Exception as e:
                logger.error(f"Error calculating faithfulness: {str(e)}")
                metrics_dict["faithfulness"] = 0.0
            
            try:
                result = evaluate(dataset, metrics=[answer_relevancy])
                metrics_dict["answer_relevancy"] = float(np.mean(result["answer_relevancy"]))
                logger.info(f"Answer relevancy calculated: {metrics_dict['answer_relevancy']}")
            except Exception as e:
                logger.error(f"Error calculating answer_relevancy: {str(e)}")
                metrics_dict["answer_relevancy"] = 0.0
            
            try:
                result = evaluate(dataset, metrics=[context_precision])
                metrics_dict["context_precision"] = float(np.mean(result["context_precision"]))
                logger.info(f"Context precision calculated: {metrics_dict['context_precision']}")
            except Exception as e:
                logger.error(f"Error calculating context_precision: {str(e)}")
                metrics_dict["context_precision"] = 0.0
            
            try:
                result = evaluate(dataset, metrics=[context_recall])
                metrics_dict["context_recall"] = float(np.mean(result["context_recall"]))
                logger.info(f"Context recall calculated: {metrics_dict['context_recall']}")
            except Exception as e:
                logger.error(f"Error calculating context_recall: {str(e)}")
                metrics_dict["context_recall"] = 0.0
            
            try:
                result = evaluate(dataset, metrics=[answer_correctness])
                metrics_dict["answer_correctness"] = float(np.mean(result["answer_correctness"]))
                logger.info(f"Answer correctness calculated: {metrics_dict['answer_correctness']}")
            except Exception as e:
                logger.error(f"Error calculating answer_correctness: {str(e)}")
                metrics_dict["answer_correctness"] = 0.0
            
            logger.info(f"RAGAS metrics calculated: {metrics_dict}")
            return {"context": context, "response": response, "metrics": metrics_dict, "next": END}
            
        except Exception as eval_error:
            logger.error(f"Error during RAGAS evaluation: {str(eval_error)}")
            logger.error(f"Error type: {type(eval_error)}")
            return {"context": context, "response": response, "metrics": {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_correctness": 0.0,
                "error": str(eval_error)
            }, "next": END}
            
    except Exception as e:
        logger.error(f"Error in RAGAS evaluation: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        return {"context": context, "response": response, "metrics": {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "answer_correctness": 0.0,
            "error": str(e)
        }, "next": END}

# Create the workflow
def create_rag_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("evaluate", evaluate_rag)
    
    # Add edges
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "evaluate")
    workflow.add_edge("evaluate", END)
    
    # Set entry point
    workflow.set_entry_point("retrieve")
    
    # Compile
    app = workflow.compile()
    return app

# Create the graph
rag_graph = create_rag_graph() 