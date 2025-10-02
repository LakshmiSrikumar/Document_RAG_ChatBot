import os
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langdetect import detect
from deep_translator import GoogleTranslator


@dataclass
class Chunk:
	text: str
	meta: Dict


class RAGPipeline:
	def __init__(self, data_dir: str = "data/index", top_k: int = 12, rerank_k: int = 6):
		self.data_dir = data_dir
		os.makedirs(self.data_dir, exist_ok=True)
		self.embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
		self.reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
		self.embed_model = None
		self.reranker = None
		self.translator = GoogleTranslator()
		self.index = None
		self.embeddings = None
		self.chunks: List[Chunk] = []
		self.top_k = top_k
		self.rerank_k = rerank_k
		self._load_models()

	def _load_models(self):
		"""Load models with error handling"""
		try:
			print("Loading embedding model...")
			self.embed_model = SentenceTransformer(self.embed_model_name)
			print("Loading reranker model...")
			self.reranker = CrossEncoder(self.reranker_model_name)
			print("Models loaded successfully!")
		except Exception as e:
			print(f"Error loading models: {e}")
			print("Using fallback models...")
			# Fallback to more reliable models
			self.embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
			self.reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
			self.embed_model = SentenceTransformer(self.embed_model_name)
			self.reranker = CrossEncoder(self.reranker_model_name)

	def translate_query(self, query: str) -> str:
		"""Translate Hindi query to English for better retrieval"""
		try:
			detected_lang = detect(query)
			if detected_lang == 'hi':  # Hindi detected
				print(f"Detected Hindi query: {query}")
				translated = self.translator.translate(query)
				print(f"Translated to English: {translated}")
				return translated
			else:
				return query  # Already in English or other language
		except Exception as e:
			print(f"Translation failed: {e}")
			return query  # Return original query if translation fails

	def load_and_chunk(self, pdf_paths: List[str]) -> List[Chunk]:
		chunks: List[Chunk] = []
		splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
		for path in pdf_paths:
			loader = PyPDFLoader(path)
			pages = loader.load()
			for p_idx, p in enumerate(pages):
				texts = splitter.split_text(p.page_content)
				for i, piece in enumerate(texts):
					if not piece.strip():
						continue
					chunks.append(Chunk(piece, {
						"filename": os.path.basename(path),
						"doc_path": path,
						"page": p_idx + 1,
						"chunk_index": i
					}))
		self.chunks.extend(chunks)
		return chunks

	def build_index(self) -> None:
		from faiss import IndexFlatIP
		if self.embed_model is None:
			print("Embedding model not loaded!")
			return
		texts = [c.text for c in self.chunks]
		if not texts:
			self.index = None
			self.embeddings = None
			return
		embs = self.embed_model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
		self.embeddings = np.array(embs, dtype=np.float32)
		dim = self.embeddings.shape[1]
		index = IndexFlatIP(dim)
		index.add(self.embeddings)
		self.index = index
		self._persist_state()

	def _persist_state(self) -> None:
		np.save(os.path.join(self.data_dir, "embeddings.npy"), self.embeddings)
		with open(os.path.join(self.data_dir, "chunks.json"), "w", encoding="utf-8") as f:
			json.dump([{"text": c.text, "meta": c.meta} for c in self.chunks], f, ensure_ascii=False)

	def load_state(self) -> bool:
		emb_path = os.path.join(self.data_dir, "embeddings.npy")
		chunks_path = os.path.join(self.data_dir, "chunks.json")
		if not (os.path.exists(emb_path) and os.path.exists(chunks_path)):
			return False
		self.embeddings = np.load(emb_path)
		with open(chunks_path, "r", encoding="utf-8") as f:
			raw = json.load(f)
		self.chunks = [Chunk(r["text"], r["meta"]) for r in raw]
		from faiss import IndexFlatIP
		dim = self.embeddings.shape[1]
		index = IndexFlatIP(dim)
		index.add(self.embeddings)
		self.index = index
		return True

	def retrieve(self, query: str) -> List[Tuple[Chunk, float]]:
		if self.index is None or self.embeddings is None or not self.chunks or self.embed_model is None:
			return []
		
		# Translate Hindi query to English for better retrieval
		translated_query = self.translate_query(query)
		
		q = self.embed_model.encode([translated_query], normalize_embeddings=True)
		scores, ids = self.index.search(np.array(q, dtype=np.float32), self.top_k)
		cand = [(self.chunks[int(i)], float(scores[0][j])) for j, i in enumerate(ids[0])]
		pairs = [[translated_query, c.text] for c, _ in cand]
		if not pairs or self.reranker is None:
			return cand[:self.rerank_k]
		rr = self.reranker.predict(pairs)
		# Keep original similarity score but order by reranker score
		ranked = sorted(list(zip(cand, rr)), key=lambda x: x[1], reverse=True)
		ordered = [c for (c, _r) in ranked][: self.rerank_k]
		return ordered

	@staticmethod
	def format_context(cands: List[Tuple[Chunk, float]]) -> str:
		blocks = []
		for (chunk, score) in cands:
			m = chunk.meta
			blocks.append(f"[{m['filename']} p.{m['page']}] {chunk.text}")
		return "\n\n".join(blocks)
