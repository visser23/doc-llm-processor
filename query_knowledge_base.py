#!/usr/bin/env python3
"""
Query utility for the processed knowledge base.
Provides search and retrieval functions for the processed documents.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from collections import defaultdict

class KnowledgeBaseQuery:
    def __init__(self, knowledge_base_path: str = "processed_knowledge_base"):
        """
        Initialize the query interface.
        
        Args:
            knowledge_base_path: Path to the processed knowledge base directory
        """
        self.kb_path = Path(knowledge_base_path)
        self.chunks = []
        self.metadata = {}
        self.file_index = {}
        
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """Load the knowledge base from files."""
        # Try to load from JSON first, then JSONL
        json_path = self.kb_path / "knowledge_base.json"
        jsonl_path = self.kb_path / "knowledge_base_chunks.jsonl"
        metadata_path = self.kb_path / "processing_metadata.json"
        
        if json_path.exists():
            print(f"Loading knowledge base from {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.chunks = data.get("chunks", [])
                self.metadata = data.get("metadata", {})
                self.file_index = data.get("file_index", {})
        elif jsonl_path.exists():
            print(f"Loading knowledge base from {jsonl_path}")
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                self.chunks = [json.loads(line) for line in f]
            
            # Load metadata separately
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                    self.metadata = meta_data.get("metadata", {})
                    self.file_index = meta_data.get("file_index", {})
        else:
            raise FileNotFoundError(f"No knowledge base found in {self.kb_path}")
        
        print(f"Loaded {len(self.chunks)} chunks from {len(self.file_index)} files")
    
    def search_text(self, query: str, max_results: int = 10, case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """
        Search for text within the knowledge base.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            List of matching chunks with relevance scores
        """
        results = []
        
        # Prepare search pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(query, flags)
        except re.error:
            # If regex fails, treat as literal string
            pattern = re.compile(re.escape(query), flags)
        
        for chunk in self.chunks:
            text = chunk["text"]
            matches = list(pattern.finditer(text))
            
            if matches:
                # Calculate relevance score based on number of matches and position
                score = len(matches)
                first_match_pos = matches[0].start() / len(text)  # Earlier matches score higher
                relevance_score = score * (1 - first_match_pos * 0.1)
                
                # Create highlighted text snippet
                snippet = self._create_snippet(text, matches[0], context_chars=200)
                
                results.append({
                    "chunk": chunk,
                    "relevance_score": relevance_score,
                    "match_count": len(matches),
                    "snippet": snippet,
                    "source_info": f"{chunk['source_file']} (chunk {chunk['chunk_index']})"
                })
        
        # Sort by relevance score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:max_results]
    
    def _create_snippet(self, text: str, match, context_chars: int = 200) -> str:
        """Create a text snippet around a match."""
        start = max(0, match.start() - context_chars // 2)
        end = min(len(text), match.end() + context_chars // 2)
        
        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        
        return snippet
    
    def get_file_content(self, filename: str) -> List[Dict[str, Any]]:
        """
        Get all chunks from a specific file.
        
        Args:
            filename: Name of the file (can be partial)
            
        Returns:
            List of chunks from matching files
        """
        matching_chunks = []
        for chunk in self.chunks:
            if filename.lower() in chunk["source_file"].lower():
                matching_chunks.append(chunk)
        
        return sorted(matching_chunks, key=lambda x: x["chunk_index"])
    
    def list_files(self, pattern: str = None) -> List[str]:
        """
        List all files in the knowledge base.
        
        Args:
            pattern: Optional pattern to filter files
            
        Returns:
            List of file paths
        """
        files = list(self.file_index.keys())
        
        if pattern:
            files = [f for f in files if pattern.lower() in f.lower()]
        
        return sorted(files)
    
    def get_file_stats(self, filename: str = None) -> Dict[str, Any]:
        """
        Get statistics about files in the knowledge base.
        
        Args:
            filename: Optional specific file to get stats for
            
        Returns:
            Dictionary with file statistics
        """
        if filename:
            # Stats for specific file
            matching_files = {k: v for k, v in self.file_index.items() 
                            if filename.lower() in k.lower()}
            return matching_files
        else:
            # Overall stats
            total_chunks = sum(info["chunk_count"] for info in self.file_index.values())
            total_chars = sum(info["total_characters"] for info in self.file_index.values())
            total_size = sum(info["file_size_bytes"] for info in self.file_index.values())
            
            return {
                "total_files": len(self.file_index),
                "total_chunks": total_chunks,
                "total_characters": total_chars,
                "total_size_bytes": total_size,
                "average_chunks_per_file": total_chunks / len(self.file_index) if self.file_index else 0,
                "processing_metadata": self.metadata
            }
    
    def export_for_llm(self, query: str = None, max_chunks: int = 50, output_file: str = None) -> str:
        """
        Export relevant chunks in a format optimized for LLM consumption.
        
        Args:
            query: Optional search query to filter relevant chunks
            max_chunks: Maximum number of chunks to include
            output_file: Optional file to save the export
            
        Returns:
            Formatted text suitable for LLM context
        """
        if query:
            search_results = self.search_text(query, max_results=max_chunks)
            chunks_to_export = [result["chunk"] for result in search_results]
        else:
            chunks_to_export = self.chunks[:max_chunks]
        
        # Format for LLM consumption
        formatted_text = "# Knowledge Base Export\n\n"
        formatted_text += f"Generated from {len(self.file_index)} source documents\n"
        formatted_text += f"Total chunks: {len(chunks_to_export)}\n\n"
        
        current_file = None
        for chunk in chunks_to_export:
            # Add file header when switching files
            if chunk["source_file"] != current_file:
                current_file = chunk["source_file"]
                formatted_text += f"\n## Source: {current_file}\n"
                if chunk["metadata"].get("page_number"):
                    formatted_text += f"Page: {chunk['metadata']['page_number']}\n"
                formatted_text += "\n"
            
            # Add chunk content
            element_type = chunk.get('element_type', 'text')
            formatted_text += f"### Chunk {chunk['chunk_index']} ({element_type})\n"
            formatted_text += f"{chunk['text']}\n\n"
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
            print(f"Exported to {output_file}")
        
        return formatted_text


def main():
    parser = argparse.ArgumentParser(description="Query the processed knowledge base")
    parser.add_argument("--kb-path", "-p", 
                       default="processed_knowledge_base",
                       help="Path to knowledge base directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for text")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--max-results", "-n", type=int, default=10,
                              help="Maximum results to return")
    search_parser.add_argument("--case-sensitive", action="store_true",
                              help="Case sensitive search")
    
    # List files command
    list_parser = subparsers.add_parser("list", help="List files")
    list_parser.add_argument("--pattern", "-p", help="Filter pattern")
    
    # File content command
    content_parser = subparsers.add_parser("content", help="Get file content")
    content_parser.add_argument("filename", help="Filename (can be partial)")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get statistics")
    stats_parser.add_argument("--file", "-f", help="Specific file stats")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export for LLM")
    export_parser.add_argument("--query", "-q", help="Search query to filter content")
    export_parser.add_argument("--max-chunks", "-n", type=int, default=50,
                              help="Maximum chunks to export")
    export_parser.add_argument("--output", "-o", help="Output file")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize query interface
    try:
        kb = KnowledgeBaseQuery(args.kb_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Execute command
    if args.command == "search":
        results = kb.search_text(args.query, args.max_results, args.case_sensitive)
        print(f"Found {len(results)} results for '{args.query}':\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['source_info']} (score: {result['relevance_score']:.2f})")
            print(f"   {result['snippet']}")
            print()
    
    elif args.command == "list":
        files = kb.list_files(args.pattern)
        print(f"Found {len(files)} files:")
        for file in files:
            info = kb.file_index[file]
            print(f"  {file} ({info['chunk_count']} chunks, {info['total_characters']} chars)")
    
    elif args.command == "content":
        chunks = kb.get_file_content(args.filename)
        if chunks:
            print(f"Content from files matching '{args.filename}':")
            current_file = None
            for chunk in chunks:
                if chunk["source_file"] != current_file:
                    current_file = chunk["source_file"]
                    print(f"\n=== {current_file} ===")
                element_type = chunk.get('element_type', 'text')
                print(f"\nChunk {chunk['chunk_index']} ({element_type}):")
                print(chunk["text"])
        else:
            print(f"No files found matching '{args.filename}'")
    
    elif args.command == "stats":
        if args.file:
            stats = kb.get_file_stats(args.file)
            print(f"Statistics for files matching '{args.file}':")
            for file, info in stats.items():
                print(f"  {file}:")
                print(f"    Chunks: {info['chunk_count']}")
                print(f"    Characters: {info['total_characters']}")
                print(f"    Size: {info['file_size_bytes']} bytes")
        else:
            stats = kb.get_file_stats()
            print("Knowledge Base Statistics:")
            print(f"  Total files: {stats['total_files']}")
            print(f"  Total chunks: {stats['total_chunks']}")
            print(f"  Total characters: {stats['total_characters']:,}")
            print(f"  Total size: {stats['total_size_bytes']:,} bytes")
            print(f"  Average chunks per file: {stats['average_chunks_per_file']:.1f}")
    
    elif args.command == "export":
        export_text = kb.export_for_llm(args.query, args.max_chunks, args.output)
        if not args.output:
            print(export_text)


if __name__ == "__main__":
    main()