"""
PDF Parser Module

Handles PDF document parsing and text extraction.
"""

import PyPDF2
import pdfplumber
from typing import List, Optional
from pathlib import Path


class PDFParser:
    """Class for parsing PDF documents and extracting text."""
    
    def __init__(self, method: str = 'pdfplumber'):
        """
        Initialize PDF parser.
        
        Args:
            method: Parsing method ('pdfplumber' or 'pypdf2')
        """
        self.method = method
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            Extracted text as string
        """
        if self.method == 'pdfplumber':
            return self._extract_with_pdfplumber(pdf_path)
        else:
            return self._extract_with_pypdf2(pdf_path)
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber (better layout preservation)."""
        text = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        
        return '\n\n'.join(text)
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2."""
        text = []
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        
        return '\n\n'.join(text)
    
    def extract_text_by_page(self, pdf_path: str) -> List[str]:
        """
        Extract text from each page separately.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            List of text strings, one per page
        """
        pages_text = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                pages_text.append(page_text if page_text else '')
        
        return pages_text
    
    def get_metadata(self, pdf_path: str) -> dict:
        """
        Extract metadata from PDF.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            Dictionary containing PDF metadata
        """
        metadata = {}
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            if reader.metadata:
                metadata = {
                    'title': reader.metadata.get('/Title', ''),
                    'author': reader.metadata.get('/Author', ''),
                    'subject': reader.metadata.get('/Subject', ''),
                    'creator': reader.metadata.get('/Creator', ''),
                    'pages': len(reader.pages)
                }
        
        return metadata
    
    def save_extracted_text(self, pdf_path: str, output_path: str) -> None:
        """
        Extract text and save to a text file.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Path for the output text file
        """
        text = self.extract_text(pdf_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)


def parse_pdf(pdf_path: str, method: str = 'pdfplumber') -> str:
    """
    Convenience function to parse a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        method: Parsing method
    
    Returns:
        Extracted text
    """
    parser = PDFParser(method=method)
    return parser.extract_text(pdf_path)
