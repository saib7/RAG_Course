# 🔬 RAG Course - Document Processing & Vector Embeddings R&D

A comprehensive **R&D project** implementing production-ready **Retrieval-Augmented Generation (RAG)** pipelines with advanced document processing, vector embeddings, and similarity analysis.

## 🎯 Project Overview

This research project provides a complete foundation for RAG applications, covering:

- **📄 Multi-Format Processing**: PDF, Word, CSV, Excel, JSON, and database integration
- **🧠 Vector Embeddings**: OpenAI and Google Gemini integration with similarity analysis  
- **⚡ Production Patterns**: Error handling, batch processing, and performance optimization
- **📚 Educational Content**: Comprehensive documentation and step-by-step learning modules

### 🔧 Core Technologies
- **Package Manager**: [UV](https://github.com/astral-sh/uv) for fast dependency management
- **AI Framework**: LangChain with OpenAI and Google Gemini embeddings
- **Document Processing**: PyMuPDF, python-docx, pandas, sqlite3
- **Analysis**: scikit-learn, SciPy, NumPy for similarity computations

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+** with **UV package manager** ([Install UV](https://docs.astral.sh/uv/getting-started/installation/))
- **API Keys**: OpenAI API key (optional: Google API key for Gemini)

### Setup & Installation
```bash
# Clone and setup environment
git clone https://github.com/saib7/RAG_Course.git
cd RAG_Course
uv sync                          # Install all dependencies
source .venv/bin/activate        # Activate environment

# Configure API keys
echo "OPENAI_API_KEY=your_key_here" > .env

# Start exploring
jupyter notebook                 # Launch Jupyter to explore notebooks
```

## 🏗️ Research Modules

```
RAG_Course/
├── 📁 1.Data_Ingest_Parsing/              # Complete document processing pipeline
│   ├── 📓 1-data_ingestion.ipynb          # ✅ Document loading fundamentals  
│   ├── 📓 2-data_parsing_pdf.ipynb        # ✅ Advanced PDF processing
│   ├── � 3-data_parsing_doc.ipynb        # ✅ Word document handling
│   ├── 📓 4-data_parsing_csv_excel.ipynb  # ✅ Structured data processing
│   ├── � 5-data_parsing_json.ipynb       # ✅ JSON/JSONL processing  
│   ├── 📓 6-data_parsing_database.ipynb   # ✅ Database integration
│   └── � data/                           # Sample datasets for all formats
├── � 2.Vector_Embedding/                 # Vector analysis & similarity
│   ├── � 1-embeddings.ipynb              # ✅ Basic embedding concepts
│   └── 📓 2-openai_gemini_embeddings.ipynb # ✅ Advanced similarity analysis
└── � 0-DataIngestion_Persing/            # Initial experiments (legacy)
```

### 🎯 Key Research Areas
- **📄 Document Processing**: Complete pipeline for PDF, Word, CSV, Excel, JSON, databases
- **🧠 Vector Embeddings**: OpenAI and Gemini integration with similarity analysis
- **⚡ Performance Analysis**: Memory usage, processing time, and optimization patterns
- **🏭 Production Ready**: Error handling, validation, and enterprise patterns

## 💻 Usage Examples

### Document Processing
```python
# Multi-format document loading
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader, JSONLoader

# Load different document types
pdf_docs = PyMuPDFLoader("1.Data_Ingest_Parsing/data/pdf/attention.pdf").load()
csv_docs = CSVLoader("1.Data_Ingest_Parsing/data/structured_files/products.csv").load()
json_docs = JSONLoader("1.Data_Ingest_Parsing/data/json_files/company_data.json").load()
```

### Vector Embeddings & Similarity Analysis
```python
# Generate embeddings and calculate similarities
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
sentence_embeddings = embeddings.embed_documents([
    "Machine learning transforms technology",
    "AI revolutionizes computing", 
    "The weather is nice today"
])

# Calculate similarity matrix
import numpy as np
similarity_matrix = cosine_similarity(np.array(sentence_embeddings))
print("Most similar:", similarity_matrix[0][1])  # Compare first two sentences
```

## 📊 Research Highlights

### Similarity Analysis Comparison
| Method | Use Case | Performance | Implementation |
|--------|----------|-------------|----------------|
| **Manual** | Learning/Education | Slow | `np.dot(v1,v2)/(norm(v1)*norm(v2))` |
| **SciPy** | Individual pairs | Medium | `1 - cosine(v1, v2)` |
| **Scikit-learn** | Batch processing | Fast | `cosine_similarity(matrix)` |

### Document Format Coverage
- **✅ PDF**: PyMuPDF with image extraction and metadata
- **✅ Word**: Docx2txt and UnstructuredWord loaders  
- **✅ Structured**: CSV/Excel with pandas integration
- **✅ JSON**: Nested structure handling and JSONL streaming
- **✅ Database**: SQLite with relationship preservation and security patterns

## �️ Development & Extension

### Environment Management
```bash
uv sync                    # Install exact dependencies from lock file
uv add package-name        # Add new packages  
uv add --dev pytest        # Add development tools
uv lock --upgrade          # Update all dependencies
```

### Adding New Research
- **New document formats**: Extend the parsing pipeline in `1.Data_Ingest_Parsing/`
- **New embedding providers**: Add to `2.Vector_Embedding/` with comparison analysis
- **Performance optimizations**: Include benchmarking and memory usage analysis
- **Educational content**: Maintain comprehensive documentation and examples

## 📄 License & Usage

This R&D project is designed for **educational and research purposes**. All notebooks include comprehensive documentation, performance analysis, and production-ready patterns.

**Key Features:**
- 🔬 **Research-focused**: Comparative analysis and performance benchmarking
- 📚 **Educational**: Step-by-step learning with detailed explanations  
- � **Production-ready**: Error handling, validation, and optimization patterns
- ⚡ **Performance-aware**: Memory usage and processing time monitoring

---

**Start exploring:** `jupyter notebook` → Navigate to `1.Data_Ingest_Parsing/1-data_ingestion.ipynb`

*Built with ❤️ using UV, LangChain, OpenAI, and modern Python practices*