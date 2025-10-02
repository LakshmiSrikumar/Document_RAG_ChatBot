#!/usr/bin/env python3
"""
Test script to verify models load correctly before running the main app.
"""

def test_model_loading():
    try:
        print("Testing model loading...")
        
        # Test embedding model
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model...")
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("✓ Embedding model loaded successfully")
        
        # Test reranker model
        from sentence_transformers import CrossEncoder
        print("Loading reranker model...")
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("✓ Reranker model loaded successfully")
        
        # Test FAISS
        import faiss
        import numpy as np
        print("Testing FAISS...")
        dim = 384  # all-MiniLM-L6-v2 dimension
        index = faiss.IndexFlatIP(dim)
        test_emb = np.random.random((1, dim)).astype(np.float32)
        index.add(test_emb)
        print("✓ FAISS working correctly")
        
        # Test translation functionality
        print("Testing translation...")
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator()
        test_hindi = "मशीन लर्निंग क्या है?"
        translated = translator.translate(test_hindi)
        print(f"✓ Translation working: '{test_hindi}' -> '{translated}'")
        
        print("\n🎉 All models loaded successfully! You can now run 'streamlit run app.py'")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have internet connection for model downloads")
        print("2. Try: pip install --upgrade sentence-transformers transformers torch")
        print("3. If still failing, try: pip install --no-cache-dir sentence-transformers")
        return False
    
    return True

if __name__ == "__main__":
    test_model_loading()
