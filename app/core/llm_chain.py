# app/core/llm_chain.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough

def create_rag_chain(vectorstore, api_key, api_base):
    """
    Builds the full RAG (Retrieval-Augmented Generation) pipeline
    using LangChain's Runnable interface.
    """

    # Step 1: Retriever - fetch relevant chunks from vectorstore
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Step 2: LLM - Use OpenRouter (free OpenAI-compatible API)
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_key=api_key,
        openai_api_base=api_base
    )

    # Step 3: Prompt template
    prompt = PromptTemplate(
        template = """
        You are a helpful AI assistant.
        Use ONLY the context provided below to answer the question.
        If the answer is not in the context, just say: "I don't know."

        Context:
        {context}

        Question:
        {question}
        """,
        input_variables=['context','question']
    )

    # Helper: format documents into plain text
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Step 4: Create parallel retriever + passthrough pipeline
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    # Step 5: Combine everything into one chain
    rag_chain = parallel_chain | prompt | llm | StrOutputParser()
    return rag_chain
