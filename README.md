# 🤖 RAG Course - Advanced Document Processing & Retrieval

A comprehensive R&D project exploring **Retrieval-Augmented Generation (RAG)** techniques using LangChain, LangGraph, and modern document processing methods.

## 📋 Table of Contents
- [🎯 Project Overview](#-project-overview)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🏗️ Project Structure](#️-project-structure)
- [💻 Usage Guide](#-usage-guide)
- [📚 Notebooks Overview](#-notebooks-overview)
- [🛠️ Development](#️-development)
- [🤝 Contributing](#-contributing)

## 🎯 Project Overview

This project demonstrates advanced **RAG (Retrieval-Augmented Generation)** implementation focusing on:

- **Document Ingestion**: PDF, Word, Excel, and web content processing
- **Smart Parsing**: Advanced text extraction with metadata enhancement
- **Vector Storage**: Efficient document embedding and retrieval
- **LangChain Integration**: Production-ready RAG pipelines
- **LangGraph Workflows**: Complex multi-step reasoning chains

### 🔧 Key Technologies
- **Package Manager**: [UV](https://github.com/astral-sh/uv) (10-100x faster than pip)
- **Framework**: LangChain, LangGraph, LangSmith
- **Vector Stores**: ChromaDB, FAISS
- **ML Libraries**: Sentence Transformers, HuggingFace
- **Document Processing**: PyMuPDF, PyPDF, python-docx

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+**
- **UV package manager** ([Install UV](https://docs.astral.sh/uv/getting-started/installation/))

### 1️⃣ Clone & Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd RAG_Course

# Install all dependencies (uses uv.lock for exact versions)
uv sync

# Activate the virtual environment
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows
```

### 2️⃣ Quick Test
```bash
# Run a sample notebook
jupyter notebook 0-DataIngestion_Persing/1-data_ingestion.ipynb
```

## 📦 Installation

### Using UV (Recommended)
```bash
# Install all project dependencies
uv sync

# Add new packages during development
uv add package-name

# Update dependencies
uv lock --upgrade
```

### Traditional Method (If needed)
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install from requirements
pip install -r requirements.txt
```

## 🏗️ Project Structure

```
RAG_Course/
├── 📁 0-DataIngestion_Persing/
│   ├── 📓 1-data_ingestion.ipynb      # Basic document loading techniques
│   ├── 📓 2-data_parsing_pdf.ipynb    # Advanced PDF processing
│   └── 📁 data/
│       ├── 📁 pdf/                    # Sample PDF documents
│       │   └── attention.pdf
│       └── 📁 text_files/             # Sample text files
│           ├── machine_learning.txt
│           └── python_intro.txt
├── 📄 main.py                         # Main application entry point
├── 📄 pyproject.toml                  # Project configuration & dependencies
├── 📄 uv.lock                         # Exact dependency versions
├── 📄 requirements.txt                # Pip compatibility
└── 📄 README.md                       # This file
```

## 💻 Usage Guide

### 🔍 Document Processing Workflow

1. **Data Ingestion**
   ```python
   # Load various document types
   from langchain_community.document_loaders import PyMuPDFLoader
   
   loader = PyMuPDFLoader("data/pdf/document.pdf")
   documents = loader.load()
   ```

2. **Smart Processing**
   ```python
   # Use advanced processors with metadata enhancement
   processor = SmartPDFProcessor2(chunk_size=800, chunk_overlap=150)
   processed_docs = processor.process_pdf("data/pdf/document.pdf")
   ```

3. **Vector Storage**
   ```python
   # Create embeddings and store in vector database
   from langchain_community.vectorstores import Chroma
   
   vectorstore = Chroma.from_documents(processed_docs, embeddings)
   ```

### 🧪 Running Experiments

```bash
# Start Jupyter for interactive development
jupyter notebook

# Run specific processing pipeline
python main.py --input data/pdf/ --output processed/

# Process single document
python -c "
from processors import SmartPDFProcessor2
processor = SmartPDFProcessor2()
docs = processor.process_pdf('data/pdf/attention.pdf')
print(f'Processed {len(docs)} chunks')
"
```

## 📚 Notebooks Overview

### 🔬 Research & Development Notebooks

| Notebook | Description | Key Features |
|----------|-------------|--------------|
| **1-data_ingestion.ipynb** | Document loading fundamentals | • Multiple loader types<br>• Basic chunking strategies<br>• Metadata handling |
| **2-data_parsing_pdf.ipynb** | Advanced PDF processing | • PyMuPDF integration<br>• Image handling<br>• Content analysis<br>• Quality assessment |

### 🎯 Learning Objectives

Each notebook includes:
- **📖 Conceptual explanations** with real-world context
- **💻 Hands-on code examples** with detailed comments
- **🔍 Comparative analysis** of different approaches
- **🛠️ Best practices** for production usage
- **❗ Common pitfalls** and troubleshooting tips

## 🛠️ Development

### 🔄 Environment Management
```bash
# Sync environment (install exact versions from uv.lock)
uv sync

# Add development dependencies
uv add --dev pytest black flake8 mypy

# Update all dependencies
uv lock --upgrade

# Check dependency tree
uv tree
```

### 🧪 Testing & Quality
```bash
# Install test dependencies
uv add --dev pytest pytest-cov

# Run tests
uv run pytest

# Code formatting
uv run black .

# Type checking
uv run mypy .
```

### 📝 Adding New Dependencies
```bash
# Add ML/AI packages
uv add transformers torch sentence-transformers

# Add development tools
uv add --dev jupyter ipykernel

# Add specific versions
uv add "langchain>=0.3.0,<0.4.0"
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file for configuration:
```bash
# API Keys
OPENAI_API_KEY=your_openai_key
HUGGINGFACE_API_TOKEN=your_hf_token

# Model Settings
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Vector Store
VECTOR_STORE_PATH=./vector_stores
```

### Custom Settings
```python
# config.py
PROCESSING_CONFIG = {
    "pdf_processor": "PyMuPDFLoader",
    "chunk_size": 800,
    "chunk_overlap": 150,
    "include_images": True,
    "quality_threshold": "medium"
}
```

## 🚀 Production Deployment

### Docker Setup
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml uv.lock ./

# Install UV and dependencies
RUN pip install uv && uv sync --frozen

COPY . .
CMD ["uv", "run", "python", "main.py"]
```

### Performance Optimization
- **UV Benefits**: 10-100x faster dependency resolution
- **Caching**: UV automatically caches packages
- **Reproducibility**: `uv.lock` ensures consistent environments
- **Memory Efficiency**: Optimized for large ML dependencies

## 🤝 Contributing

### Development Workflow
1. **Fork & Clone**: Get your own copy
2. **Environment**: `uv sync` to install dependencies
3. **Branch**: Create feature branch
4. **Develop**: Make changes with proper documentation
5. **Test**: Ensure all notebooks run successfully
6. **PR**: Submit with clear description

### Code Standards
- **Documentation**: Comprehensive docstrings and comments
- **Notebooks**: Clear explanations and educational content
- **Dependencies**: Use `uv add` for new packages
- **Commits**: Descriptive commit messages

## 📄 License

This project is for educational and research purposes. See course materials for specific licensing terms.

## 🆘 Support

### Common Issues
- **UV Installation**: [Official UV docs](https://docs.astral.sh/uv/)
- **Dependency Conflicts**: `uv sync --refresh`
- **Notebook Kernel**: `uv run python -m ipykernel install --user`

### Getting Help
- **Documentation**: Check notebook explanations first
- **Dependencies**: Review `pyproject.toml` for version info
- **Performance**: UV provides detailed timing information

---

**Happy Learning! 🎓**

*Built with ❤️ using UV, LangChain, and modern Python practices*