# GenerativeAI-LLM-RAG

Foundation Models and RAG Labs
A collection of end-to-end notebooks exploring decoder-only Transformers from scratch, BERT fine-tuning across frameworks, multi-model fine-tuning with Hugging Face, and practical semantic search/RAG pipelines with ChromaDB and Weaviate.

What’s inside:- 
-mini-transformer-decoder.ipynb
Builds a GPT-style, decoder-only Transformer from scratch: scaled dot-product attention, multi-head attention, sinusoidal positional encodings, masked self-attention blocks, a minimal training loop on toy text, and attention map visualizations.

-BERT_TensorFlow_vs_HuggingFace_Comparison.ipynb
Implements BERT text classification with both TensorFlow Hub and Hugging Face, using the same dataset/splits/epochs, then compares accuracy, macro-F1, confusion matrices, runtime, GPU memory, sequence lengths, and implementation effort.

-Finetune_BERT_GPT2_BART.ipynb
Loads and fine-tunes three families with Hugging Face: BERT for sentiment classification (IMDB), GPT-2 for language modeling and generation (Wikitext-2), and BART for summarization (CNN/DailyMail); includes short demo training and inference pipelines.

-LangChain_Primitives_and_JSON_Parsing.ipynb
Sets up an LLM API key flow, demonstrates LangChain model primitives and PromptTemplates, builds prompt→model chains, parses plain text and structured JSON (JsonOutputParser, PydanticOutputParser, OutputFixingParser), and contrasts temperature 0.1 vs 0.9 outputs.

-chromadb_weaviate_semantic_search.ipynb
Two semantic search tracks over the same mini corpus: ChromaDB local collection with sentence-transformer embeddings and cosine distance; Weaviate Cloud collection with manual vectorization, HNSW indexing, and semantic queries; includes multi-query evaluation.

-text_chunking_hnsw_rag_pipeline.ipynb
End-to-end RAG lab featuring multiple chunking strategies (fixed, sentence, paragraph, sliding window, heading-based, semantic, hybrid), FAISS HNSW retrieval, a small embedding model, a simple LLM client, and an interactive RAG loop with qualitative comparisons.

