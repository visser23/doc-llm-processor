#!/usr/bin/env python3
"""
Document Processing Pipeline using Unstructured.io
Processes various document types and creates a knowledge base suitable for LLMs.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm
import hashlib
from datetime import datetime

# Unstructured imports
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import dict_to_elements, elements_to_json

class DocumentProcessor:
    def __init__(self, 
                 input_dir: str, 
                 output_dir: str = "processed_knowledge_base",
                 chunk_size: int = 1000,
                 overlap: int = 200,
                 resume: bool = True):
        """
        Initialize the document processor.
        
        Args:
            input_dir: Directory containing documents to process
            output_dir: Directory to save processed output
            chunk_size: Maximum characters per chunk
            overlap: Character overlap between chunks
            resume: Whether to resume from existing progress
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.resume = resume
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Supported file extensions
        self.supported_extensions = {
            '.pdf', '.docx', '.doc', '.pptx', '.ppt', 
            '.xlsx', '.xls', '.txt', '.md', '.html', '.htm'
        }
        
        # Track processing results
        self.processed_files = []
        self.failed_files = []
        self.total_chunks = 0
        self.existing_chunks = []
        
        # Load existing progress if resuming
        if self.resume:
            self._load_existing_progress()
        
    def get_file_hash(self, file_path: Path) -> str:
        """Generate a hash for the file to track changes."""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _load_existing_progress(self):
        """Load existing processed chunks to enable resuming."""
        progress_file = self.output_dir / "progress.json"
        if progress_file.exists():
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    self.existing_chunks = progress_data.get("chunks", [])
                    self.processed_files = progress_data.get("processed_files", [])
                    print(f"Loaded {len(self.existing_chunks)} existing chunks from {len(self.processed_files)} files")
            except Exception as e:
                print(f"Could not load existing progress: {e}")
                self.existing_chunks = []
    
    def _save_progress(self, current_chunks):
        """Save current progress to enable resuming."""
        progress_file = self.output_dir / "progress.json"
        progress_data = {
            "chunks": current_chunks,
            "processed_files": self.processed_files,
            "timestamp": datetime.now().isoformat()
        }
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)
    
    def _is_file_processed(self, file_path: Path) -> bool:
        """Check if a file has already been processed."""
        file_str = str(file_path.relative_to(self.input_dir))
        return file_str in self.processed_files
    
    def find_documents(self) -> List[Path]:
        """Find all supported documents in the input directory."""
        documents = []
        for ext in self.supported_extensions:
            documents.extend(self.input_dir.rglob(f"*{ext}"))
        return sorted(documents)
    
    def process_single_document(self, file_path: Path) -> Optional[List[Dict[str, Any]]]:
        """
        Process a single document and return structured chunks.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of processed chunks with metadata
        """
        try:
            # Check if file already processed
            if self.resume and self._is_file_processed(file_path):
                print(f"Skipping already processed: {file_path.name}")
                return []
            
            print(f"Processing: {file_path.name} ({file_path.stat().st_size / 1024 / 1024:.1f} MB)")
            
            # Use faster strategy for large files
            file_size_mb = file_path.stat().st_size / 1024 / 1024
            strategy = "fast" if file_size_mb > 10 else "hi_res"
            
            # Partition the document with timeout handling
            elements = partition(
                filename=str(file_path),
                strategy=strategy,
                include_page_breaks=True,
                infer_table_structure=True if file_size_mb < 5 else False,  # Skip table inference for large files
                chunking_strategy="by_title",
                max_characters=self.chunk_size,
                new_after_n_chars=self.chunk_size - self.overlap,
                combine_text_under_n_chars=100
            )
            
            # Convert to structured format
            chunks = []
            for i, element in enumerate(elements):
                chunk_data = {
                    "chunk_id": f"{file_path.stem}_{i:04d}",
                    "source_file": str(file_path.relative_to(self.input_dir)),
                    "file_hash": self.get_file_hash(file_path),
                    "chunk_index": i,
                    "element_type": str(type(element).__name__),
                    "text": element.text,
                    "metadata": {
                        "filename": file_path.name,
                        "file_directory": str(file_path.parent.relative_to(self.input_dir)),
                        "file_size_bytes": file_path.stat().st_size,
                        "processed_timestamp": datetime.now().isoformat(),
                        "page_number": getattr(element.metadata, 'page_number', None) if hasattr(element, 'metadata') else None,
                        "coordinates": getattr(element.metadata, 'coordinates', None) if hasattr(element, 'metadata') else None,
                        "parent_id": getattr(element.metadata, 'parent_id', None) if hasattr(element, 'metadata') else None,
                        "category": getattr(element.metadata, 'category', None) if hasattr(element, 'metadata') else None
                    }
                }
                
                # Only add chunks with meaningful content
                if chunk_data["text"] and len(chunk_data["text"].strip()) > 10:
                    chunks.append(chunk_data)
            
            self.processed_files.append(str(file_path))
            return chunks
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            self.failed_files.append({"file": str(file_path), "error": str(e)})
            return None
    
    def process_all_documents(self) -> Dict[str, Any]:
        """
        Process all documents in the input directory.
        
        Returns:
            Dictionary containing all processed chunks and metadata
        """
        documents = self.find_documents()
        print(f"Found {len(documents)} documents to process")
        
        # Start with existing chunks if resuming
        all_chunks = self.existing_chunks.copy() if self.resume else []
        
        # Filter out already processed documents
        if self.resume:
            remaining_docs = [doc for doc in documents if not self._is_file_processed(doc)]
            print(f"Resuming: {len(remaining_docs)} documents remaining to process")
            documents = remaining_docs
        
        # Process each document with progress bar
        for i, doc_path in enumerate(tqdm(documents, desc="Processing documents")):
            try:
                chunks = self.process_single_document(doc_path)
                if chunks:
                    all_chunks.extend(chunks)
                    self.total_chunks += len(chunks)
                
                # Save progress every 5 files
                if (i + 1) % 5 == 0:
                    self._save_progress(all_chunks)
                    
            except KeyboardInterrupt:
                print(f"\nProcessing interrupted. Saving progress...")
                self._save_progress(all_chunks)
                raise
            except Exception as e:
                print(f"Error processing {doc_path}: {str(e)}")
                self.failed_files.append({"file": str(doc_path), "error": str(e)})
                continue
        
        # Create comprehensive knowledge base structure
        knowledge_base = {
            "metadata": {
                "total_documents": len(documents),
                "successfully_processed": len(self.processed_files),
                "failed_documents": len(self.failed_files),
                "total_chunks": self.total_chunks,
                "processing_timestamp": datetime.now().isoformat(),
                "input_directory": str(self.input_dir),
                "chunk_size": self.chunk_size,
                "overlap": self.overlap
            },
            "chunks": all_chunks,
            "failed_files": self.failed_files,
            "file_index": self._create_file_index(all_chunks)
        }
        
        return knowledge_base
    
    def _create_file_index(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create an index of files for quick lookup."""
        file_index = {}
        for chunk in chunks:
            source_file = chunk["source_file"]
            if source_file not in file_index:
                file_index[source_file] = {
                    "chunk_count": 0,
                    "total_characters": 0,
                    "file_directory": chunk["metadata"]["file_directory"],
                    "filename": chunk["metadata"]["filename"],
                    "file_size_bytes": chunk["metadata"]["file_size_bytes"]
                }
            
            file_index[source_file]["chunk_count"] += 1
            file_index[source_file]["total_characters"] += len(chunk["text"])
        
        return file_index
    
    def save_knowledge_base(self, knowledge_base: Dict[str, Any], format_type: str = "json"):
        """
        Save the processed knowledge base to files.
        
        Args:
            knowledge_base: The processed knowledge base
            format_type: Output format ('json', 'jsonl', or 'both')
        """
        if format_type in ["json", "both"]:
            # Save as single JSON file
            json_path = self.output_dir / "knowledge_base.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
            print(f"Saved knowledge base to: {json_path}")
        
        if format_type in ["jsonl", "both"]:
            # Save chunks as JSONL for streaming/incremental loading
            jsonl_path = self.output_dir / "knowledge_base_chunks.jsonl"
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for chunk in knowledge_base["chunks"]:
                    f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            print(f"Saved chunks to: {jsonl_path}")
        
        # Save metadata separately
        metadata_path = self.output_dir / "processing_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": knowledge_base["metadata"],
                "file_index": knowledge_base["file_index"],
                "failed_files": knowledge_base["failed_files"]
            }, f, indent=2, ensure_ascii=False)
        print(f"Saved metadata to: {metadata_path}")
        
        # Create a summary CSV for easy overview
        self._create_summary_csv(knowledge_base)
    
    def _create_summary_csv(self, knowledge_base: Dict[str, Any]):
        """Create a CSV summary of processed files."""
        summary_data = []
        for file_path, info in knowledge_base["file_index"].items():
            summary_data.append({
                "source_file": file_path,
                "filename": info["filename"],
                "directory": info["file_directory"],
                "chunk_count": info["chunk_count"],
                "total_characters": info["total_characters"],
                "file_size_bytes": info["file_size_bytes"]
            })
        
        df = pd.DataFrame(summary_data)
        csv_path = self.output_dir / "processing_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved summary to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Process documents for LLM knowledge base")
    parser.add_argument("--input-dir", "-i", 
                       required=True,
                       help="Input directory containing documents")
    parser.add_argument("--output-dir", "-o", 
                       default="processed_knowledge_base",
                       help="Output directory for processed files")
    parser.add_argument("--chunk-size", "-c", 
                       type=int, default=1000,
                       help="Maximum characters per chunk")
    parser.add_argument("--overlap", 
                       type=int, default=200,
                       help="Character overlap between chunks")
    parser.add_argument("--format", "-f", 
                       choices=["json", "jsonl", "both"], 
                       default="both",
                       help="Output format")
    parser.add_argument("--no-resume", 
                       action="store_true",
                       help="Start fresh instead of resuming from existing progress")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = DocumentProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        resume=not args.no_resume
    )
    
    # Process documents
    print("Starting document processing...")
    knowledge_base = processor.process_all_documents()
    
    # Save results
    processor.save_knowledge_base(knowledge_base, args.format)
    
    # Print summary
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    print(f"Total documents found: {knowledge_base['metadata']['total_documents']}")
    print(f"Successfully processed: {knowledge_base['metadata']['successfully_processed']}")
    print(f"Failed to process: {knowledge_base['metadata']['failed_documents']}")
    print(f"Total chunks created: {knowledge_base['metadata']['total_chunks']}")
    print(f"Output directory: {args.output_dir}")
    
    if knowledge_base['failed_files']:
        print(f"\nFailed files:")
        for failed in knowledge_base['failed_files']:
            print(f"  - {failed['file']}: {failed['error']}")


if __name__ == "__main__":
    main()