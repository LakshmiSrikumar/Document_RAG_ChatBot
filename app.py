import os
import io
import time
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv
from langdetect import detect
from openai import OpenAI

from rag_utils import RAGPipeline, Chunk

load_dotenv()

st.set_page_config(page_title="PDF RAG (EN/HI)")
st.title("PDF RAG Chatbot (English Text Upload Only)")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "12"))
RAG_RERANK_K = int(os.getenv("RAG_RERANK_K", "6"))
SIMILARITY_ABSTAIN_THRESHOLD = float(os.getenv("SIMILARITY_ABSTAIN_THRESHOLD", "0.25"))

if "pipeline" not in st.session_state:
	st.session_state.pipeline = RAGPipeline(top_k=RAG_TOP_K, rerank_k=RAG_RERANK_K)
	st.session_state.pipeline.load_state()

if "uploaded_files_set" not in st.session_state:
	st.session_state.uploaded_files_set = set()

lang = st.radio("Response language", options=["en", "hi"], format_func=lambda x: "English" if x == "en" else "हिन्दी", horizontal=True)

uploaded_files = st.file_uploader("Upload English PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
	# Check if these are new files
	new_files = []
	for uf in uploaded_files:
		if uf.name not in st.session_state.uploaded_files_set:
			new_files.append(uf)
	
	if new_files:
		paths: List[str] = []
		os.makedirs("uploads", exist_ok=True)
		for uf in new_files:
			p = os.path.join("uploads", uf.name)
			with open(p, "wb") as f:
				f.write(uf.read())
			paths.append(p)
			st.session_state.uploaded_files_set.add(uf.name)
		
		with st.spinner("Ingesting and indexing..."):
			new_chunks = st.session_state.pipeline.load_and_chunk(paths)
			st.session_state.pipeline.build_index()
		st.success(f"Ingested {len(new_chunks)} chunks from {len(new_files)} new files. Index updated.")
	else:
		st.info("All files already processed. No re-indexing needed.")

# Show indexed files status
if st.session_state.uploaded_files_set:
	st.caption(f"📚 Indexed files: {', '.join(st.session_state.uploaded_files_set)}")

st.caption("Ask in English or Hindi. The reply will be only in your selected language.")
q = st.text_input("Your question:")

col1, col2, col3 = st.columns([1,1,1])
with col1:
	ask = st.button("Ask")
with col2:
	clear = st.button("Clear")
with col3:
	clear_index = st.button("Clear Index")

if clear:
	st.session_state.pop("last_answer", None)
	st.session_state.pop("last_sources", None)

if clear_index:
	st.session_state.uploaded_files_set.clear()
	st.session_state.pipeline.chunks.clear()
	st.session_state.pipeline.index = None
	st.session_state.pipeline.embeddings = None
	st.session_state.pop("last_answer", None)
	st.session_state.pop("last_sources", None)
	st.success("Index cleared. Upload new files to rebuild.")

client = None
if OPENAI_API_KEY:
	client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_TEMPLATE = (
	"You are a helpful assistant that ONLY uses the provided context.\n"
	"If the answer is not present, say so. Cite as (filename, page).\n"
	"Respond ONLY in: {lang}.\n"
)


def generate_answer(question: str, target_lang: str, cands: List[Tuple[Chunk, float]]) -> str:
	context = RAGPipeline.format_context(cands)
	system = SYSTEM_TEMPLATE.format(lang="English" if target_lang == "en" else "Hindi")
	prompt = f"{system}\n\nUser question:\n{question}\n\nContext:\n{context}\n\nAnswer:"
	if client is None:
		return "[LLM not configured. Set OPENAI_API_KEY in .env]"
	resp = client.chat.completions.create(
		model=OPENAI_MODEL,
		messages=[
			{"role": "system", "content": system},
			{"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
		],
		temperature=0.2,
	)
	return resp.choices[0].message.content.strip()


def enforce_output_language(text: str, target_lang: str) -> str:
	# If text already in target language, return; else ask the LLM to translate without changing meaning
	try:
		pred_lang = detect(text)
	except Exception:
		pred_lang = None
	if (target_lang == "en" and pred_lang == "en") or (target_lang == "hi" and pred_lang == "hi"):
		return text
	if client is None:
		return text
	instruction = (
		"Translate the following to natural {lng} without adding or removing information. "
		"Preserve citations like (filename, page)."
	).format(lng="English" if target_lang == "en" else "Hindi")
	resp = client.chat.completions.create(
		model=OPENAI_MODEL,
		messages=[
			{"role": "system", "content": instruction},
			{"role": "user", "content": text},
		],
		temperature=0.0,
	)
	return resp.choices[0].message.content.strip()


if ask and q:
	if st.session_state.pipeline.index is None:
		st.warning("Please upload PDFs first.")
	else:
		# Show translation info if needed
		try:
			from langdetect import detect
			detected_lang = detect(q)
			if detected_lang == 'hi':
				st.info(f"🔍 Detected Hindi query. Translating for better retrieval...")
		except:
			pass
		
		with st.spinner("Retrieving context..."):
			cands = st.session_state.pipeline.retrieve(q)
		if not cands:
			st.write("No relevant content found in the uploaded PDFs." if lang == "en" else "अपलोड किए गए PDFs में प्रासंगिक सामग्री नहीं मिली।")
		else:
			best_score = max([score for (_c, score) in cands]) if cands else 0.0
			if best_score < SIMILARITY_ABSTAIN_THRESHOLD:
				st.write("I could not find a grounded answer in the provided documents." if lang == "en" else "दिए गए दस्तावेज़ों में आधारित उत्तर नहीं मिला।")
			else:
				answer = generate_answer(q, lang, cands)
				answer = enforce_output_language(answer, lang)
				st.session_state.last_answer = answer
				st.session_state.last_sources = cands

if st.session_state.get("last_answer"):
	st.markdown(st.session_state.last_answer)
	with st.expander("Sources"):
		for (chunk, score) in st.session_state.last_sources or []:
			m = chunk.meta
			st.markdown(f"- {m['filename']} (page {m['page']})")
