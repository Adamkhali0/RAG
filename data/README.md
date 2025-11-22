# Data Directory

This directory should contain your PDF documents for indexation.

## 📁 Structure

Place your PDF files directly in this directory:

```
data/
├── document1.pdf
├── document2.pdf
├── document3.pdf
└── document4.pdf
```

## 📋 Guidelines

### Document Requirements
- **Format:** PDF files (`.pdf`)
- **Content:** Text-based PDFs (not scanned images without OCR)
- **Theme:** Documents should ideally have a common theme or subject
- **Language:** French or English (models support both)

### Recommended Topics
Choose documents on a common theme for best results:

- **Legal documents:** Laws, regulations, court decisions
- **Research papers:** AI, machine learning, data science
- **Technical documentation:** Software manuals, specifications
- **Educational content:** Course materials, textbooks
- **Business documents:** Reports, policies, procedures

### Best Practices

1. **Document Quality**
   - Use clean, well-formatted PDFs
   - Avoid scanned documents (or use OCR first)
   - Ensure text is extractable (not images)

2. **Document Size**
   - 3-4 documents minimum recommended
   - No strict maximum, but more documents = longer indexing time
   - Each document can be any length

3. **File Naming**
   - Use descriptive filenames: `ai_research_paper_2024.pdf`
   - Avoid special characters in filenames
   - No spaces preferred (use underscores or hyphens)

## 🔍 Example Document Sets

### Option 1: AI Research Papers
- Download 3-4 research papers on AI/ML from arXiv
- Topics: deep learning, NLP, computer vision
- Example: https://arxiv.org/

### Option 2: Legal Documents
- French law texts (Code civil, Code pénal)
- Court decisions
- Regulatory documents

### Option 3: Technical Documentation
- Software documentation
- API references
- Technical specifications

## 🚀 Quick Start

1. **Download PDFs:** Find and download 3-4 PDF documents on your chosen topic
2. **Place in data/:** Move them to this directory
3. **Index:** Run `python cli.py index`
4. **Query:** Run `python cli.py search "your query"`

## ⚠️ Important Notes

- **Copyright:** Ensure you have the right to use the documents
- **Privacy:** Don't include sensitive or confidential documents
- **Size:** Large files will take longer to process
- **Backup:** Keep original copies of your documents

## 📊 After Indexing

Once indexed, the system will:
- Extract text from each page
- Split into chunks (~1000 characters)
- Generate embeddings for each chunk
- Store in ChromaDB vector database

You can then:
- Search for relevant passages
- Ask questions about the content
- Get AI-generated answers with sources

## 🆘 Troubleshooting

### "No documents found"
- Check that PDF files are in this directory
- Verify file extensions are `.pdf`

### "Unable to extract text"
- PDF may be image-based (needs OCR)
- Try with a different PDF

### "Indexing takes too long"
- Normal for large documents
- Reduce chunk_size in config.yaml
- Use fewer/smaller documents

---

**Ready to start?** Place your PDFs here and run:
```bash
python cli.py index
```
