# LangGraph RAG + RAGAS

Built this RAG system to experiment with LangGraph's workflow capabilities and RAGAS metrics. It's a straightforward implementation that lets you upload docs, ask questions, and get quality metrics for each response.

## What it does

- Takes your docs and chunks them semantically (sentence-level similarity with greedy paragraph grouping)
- Uses ChromaDB to store and retrieve relevant context
- Spits out responses with RAGAS metrics:
  - Faithfulness: How well the response sticks to the context
  - Answer Relevancy: How relevant the answer is to the question
  - Context Precision: How precise the retrieved context is
  - Context Recall: How much relevant context was retrieved
  - Answer Correctness: How accurate the answer is
- Simple Streamlit UI to interact with it all

## Getting it running

1. Clone the repo
2. Install the deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Toss your OpenAI key in `.env`:
   ```
   OPENAI_API_KEY=your_key_here
   ```

## Using it

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Upload your docs in the sidebar
3. Fire away with questions
4. Check the metrics to see how well it's doing

## Under the hood

- LangGraph handles the RAG pipeline (retrieve -> generate -> evaluate)
- ChromaDB stores the vectors with cosine similarity
- GPT-3.5-turbo generates responses
- RAGAS evaluates response quality
- Streamlit for the UI

## Heads up

- Vectors get stored in `chroma_db`
- Using semantic chunking with sentence-level similarity and paragraph grouping
- Each response comes with its RAGAS metrics
- Minimum chunk size is a single sentence, max is 1000 chars 