# qa_pipeline.py

# This module contains the main logic for the question-answering pipeline.
# It brings together the data extraction, indexing, and LLM generation.

from indexer import VectorDB

class CognifyAIPipeline:
    """
    The main class for the CognifyAI system.
    """
    def __init__(self, knowledge_base: VectorDB):
        """
        Initializes the pipeline with a pre-built knowledge base.

        Args:
            knowledge_base (VectorDB): The vector database containing the video's content.
        """
        self.knowledge_base = knowledge_base
        # In a real app, you would initialize the Gemini model here.
        # e.g., genai.configure(api_key=GEMINI_API_KEY)
        # self.llm = genai.GenerativeModel('gemini-pro')
        print("[INFO] CognifyAI QA Pipeline initialized.")

    def build_prompt(self, question: str, context_chunks: list[str]) -> str:
        """
        Constructs the detailed prompt to be sent to the LLM.
        This is the core of the Retrieval-Augmented Generation (RAG) process.

        Args:
            question (str): The user's original question.
            context_chunks (list[str]): Relevant facts retrieved from the vector DB.

        Returns:
            str: The final, complete prompt for the LLM.
        """
        print("[INFO] Building the final prompt for the LLM...")
        
        context_str = "\n".join(context_chunks)
        
        prompt_template = f"""
        You are an expert AI Learning Companion. Answer the user's question based on the
        provided context from a lecture video.

        CONTEXT FROM VIDEO:
        ---
        {context_str}
        ---

        USER QUESTION:
        "{question}"

        INSTRUCTIONS:
        If the question is factual, answer using only the context. If the user asks for
        a deeper explanation, use your own knowledge to elaborate on the context.
        """
        return prompt_template.strip()

    def ask(self, question: str) -> str:
        """
        The main method to ask a question to the pipeline.

        Args:
            question (str): The user's question.

        Returns:
            str: The LLM's generated answer.
        """
        # 1. Retrieve relevant context from the knowledge base.
        retrieved_context = self.knowledge_base.search(question)
        
        # 2. Build the prompt.
        final_prompt = self.build_prompt(question, retrieved_context)
        
        # 3. Generate the answer using the LLM.
        print("[INFO] Sending the prompt to the Generative LLM (Simulation)...")
        # In a real implementation, you would make an API call here.
        # response = self.llm.generate_content(final_prompt)
        # return response.text
        
        # --- SIMULATED RESPONSE ---
        simulated_response = (
            "Based on the video, Ohm's Law (V=IR) is the fundamental relationship between "
            "voltage, current, and resistance, and it's critical for circuit design. "
            "As an analogy, think of voltage as the water pressure in a pipe, current as "
            "the flow rate of the water, and resistance as the pipe's narrowness. "
            "Ohm's law just describes how these three things are related!"
        )
        return simulated_response
