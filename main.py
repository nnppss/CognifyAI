# main.py

# This is the main entry point to run a demonstration of the CognifyAI pipeline.
# It simulates the entire process from video processing to answering questions.

import data_extractor
import indexer
from qa_pipeline import CognifyAIPipeline
from config import DEMO_VIDEO_URL

def run_demonstration():
    """
    Executes the full, simulated workflow of the CognifyAI project.
    """
    print("--- Starting CognifyAI Demonstration ---")
    
    # --- STAGE 1: DATA EXTRACTION ---
    print("\n[STAGE 1/3] Extracting data from the video source...")
    captions = data_extractor.get_captions(DEMO_VIDEO_URL)
    # In a real app, you would download the video first to a path.
    video_file_path = "downloads/demo_video.mp4" 
    key_frame_paths = data_extractor.get_key_frames(video_file_path)
    ocr_texts = [data_extractor.get_ocr_text(path) for path in key_frame_paths]
    
    # --- STAGE 2: INDEXING ---
    print("\n[STAGE 2/3] Building the knowledge base...")
    all_text_content = [captions] + ocr_texts
    
    # Initialize and build the vector database.
    db = indexer.VectorDB()
    db.build_index(all_text_content)
    
    # --- STAGE 3: QUESTION ANSWERING ---
    print("\n[STAGE 3/3] Initializing the QA Pipeline and asking questions...")
    pipeline = CognifyAIPipeline(knowledge_base=db)
    
    # Ask a factual question.
    question1 = "What did the professor say about Ohm's Law?"
    print(f"\n> USER ASKS: \"{question1}\"")
    answer1 = pipeline.ask(question1)
    print(f"< COGNIFYAI ANSWERS: \n\"{answer1}\"")
    
    # Ask an explanatory question.
    question2 = "Can you give me an analogy for Ohm's Law?"
    print(f"\n> USER ASKS: \"{question2}\"")
    answer2 = pipeline.ask(question2)
    print(f"< COGNIFYAI ANSWERS: \n\"{answer2}\"")
    
    print("\n--- Demonstration Finished ---")

if __name__ == "__main__":
    run_demonstration()
