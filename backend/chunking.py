import re
import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class LegalChunk:
    """Represents a semantically meaningful legal text chunk"""
    chunk_id: str
    chunk_type: str
    content: str
    metadata: Dict[str, Any]
    
    def to_dict(self):
        return asdict(self)


class LegalChunker:
    """
    Legal-aware document chunking that preserves judicial reasoning structure.
    Uses pattern matching to detect section types and groups related paragraphs.
    """
    
    # Judicial section patterns (case-insensitive)
    SECTION_PATTERNS = {
        'case_metadata': r'(IN THE .+ COURT|CASE NO\.|PETITION NO\.|BEFORE THE|CORAM[:;])',
        'parties': r'(PETITIONER|RESPONDENT|APPELLANT|DEFENDANT|PLAINTIFF)(?:\s*[:;]|\s+vs\.?|\s+v\.)',
        'facts': r'(FACTS OF THE CASE|BRIEF FACTS|FACTUAL BACKGROUND|BACKGROUND|CIRCUMSTANCES)',
        'issues': r'(ISSUES? (?:FOR CONSIDERATION|FRAMED|RAISED)|POINTS? FOR DETERMINATION|QUESTIONS? OF LAW)',
        'arguments_petitioner': r'(ARGUMENTS? (?:OF|BY) (?:THE )?PETITIONER|SUBMISSIONS? (?:OF|BY) (?:THE )?APPELLANT|PETITIONER\'?S CASE)',
        'arguments_respondent': r'(ARGUMENTS? (?:OF|BY) (?:THE )?RESPONDENT|SUBMISSIONS? (?:OF|BY) (?:THE )?RESPONDENT|RESPONDENT\'?S CASE)',
        'evidence': r'(EVIDENCE|PRECEDENTS? (?:CITED|CONSIDERED|RELIED)|CASE LAW|LEGAL AUTHORITIES)',
        'reasoning': r'((?:COURT\'?S )?(?:ANALYSIS|REASONING|OBSERVATIONS?|FINDINGS?)|DISCUSSION|CONSIDERATION OF ISSUES?)',
        'ratio': r'(RATIO DECIDENDI|LEGAL PRINCIPLE|HELD THAT|HOLDING)',
        'decision': r'((?:FINAL )?(?:ORDER|DECISION|JUDGMENT|DECREE)|DISPOSED? OF|IT IS (?:HEREBY )?ORDERED)'
    }
    
    def __init__(self, target_chunk_size: int = 600, overlap: int = 100):
        """
        Initialize chunker with target size constraints.
        
        Args:
            target_chunk_size: Target tokens per chunk (approx 4 chars = 1 token)
            overlap: Overlap tokens between chunks for continuity
        """
        self.target_chunk_size = target_chunk_size * 4  # Convert to characters
        self.overlap = overlap * 4
        
    def detect_section_type(self, text: str) -> str:
        """
        Detect the judicial section type using pattern matching.
        
        Args:
            text: Text content to classify
            
        Returns:
            Section type identifier
        """
        text_upper = text.upper()
        
        # Check each pattern
        for section_type, pattern in self.SECTION_PATTERNS.items():
            if re.search(pattern, text_upper):
                return section_type
                
        return 'general_content'
    
    def extract_paragraph_number(self, text: str) -> str:
        """
        Extract paragraph number if present (e.g., "12.", "[5]", "Para 3")
        
        Args:
            text: Paragraph text
            
        Returns:
            Paragraph number or empty string
        """
        # Match patterns like "12.", "[5]", "Para 3", etc.
        match = re.match(r'^\s*(?:\[?(\d+)\]?\.?|Para\.?\s*(\d+))', text)
        if match:
            return match.group(1) or match.group(2)
        return ""
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs, preserving structure.
        
        Args:
            text: Full document text
            
        Returns:
            List of paragraph strings
        """
        # Split on double newlines or numbered paragraphs
        paragraphs = re.split(r'\n\s*\n+|\n(?=\d+\.|\[\d+\])', text)
        
        # Clean and filter empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def group_paragraphs_by_section(self, paragraphs: List[str]) -> List[Dict[str, Any]]:
        """
        Group consecutive paragraphs that belong to the same judicial section.
        
        Args:
            paragraphs: List of paragraph strings
            
        Returns:
            List of section groups with metadata
        """
        sections = []
        current_section = {
            'type': 'general_content',
            'paragraphs': [],
            'start_para': None,
            'end_para': None
        }
        
        for i, para in enumerate(paragraphs):
            section_type = self.detect_section_type(para)
            para_num = self.extract_paragraph_number(para)
            
            # Start new section if type changes
            if section_type != current_section['type'] and current_section['paragraphs']:
                sections.append(current_section.copy())
                current_section = {
                    'type': section_type,
                    'paragraphs': [],
                    'start_para': None,
                    'end_para': None
                }
            
            # Update section type if detected
            if section_type != 'general_content':
                current_section['type'] = section_type
            
            # Track paragraph range
            if para_num and current_section['start_para'] is None:
                current_section['start_para'] = para_num
            if para_num:
                current_section['end_para'] = para_num
                
            current_section['paragraphs'].append(para)
        
        # Add final section
        if current_section['paragraphs']:
            sections.append(current_section)
            
        return sections
    
    def create_chunks_from_section(self, section: Dict[str, Any], case_name: str) -> List[LegalChunk]:
        """
        Create appropriately sized chunks from a section, respecting size constraints.
        
        Args:
            section: Section dictionary with paragraphs
            case_name: Name of the case for metadata
            
        Returns:
            List of LegalChunk objects
        """
        chunks = []
        current_content = []
        current_length = 0
        chunk_counter = 0
        
        for para in section['paragraphs']:
            para_length = len(para)
            
            # If adding this paragraph exceeds target, finalize current chunk
            if current_content and (current_length + para_length) > self.target_chunk_size:
                chunk_id = f"{section['type']}_{chunk_counter}"
                para_range = f"{section['start_para']}-{section['end_para']}" if section['start_para'] else "N/A"
                
                chunks.append(LegalChunk(
                    chunk_id=chunk_id,
                    chunk_type=section['type'],
                    content='\n\n'.join(current_content),
                    metadata={
                        'case_name': case_name,
                        'paragraph_range': para_range,
                        'section_type': section['type']
                    }
                ))
                
                chunk_counter += 1
                
                # Start new chunk with overlap (last paragraph)
                if self.overlap > 0 and current_content:
                    current_content = [current_content[-1]]
                    current_length = len(current_content[0])
                else:
                    current_content = []
                    current_length = 0
            
            current_content.append(para)
            current_length += para_length
        
        # Add remaining content as final chunk
        if current_content:
            chunk_id = f"{section['type']}_{chunk_counter}"
            para_range = f"{section['start_para']}-{section['end_para']}" if section['start_para'] else "N/A"
            
            chunks.append(LegalChunk(
                chunk_id=chunk_id,
                chunk_type=section['type'],
                content='\n\n'.join(current_content),
                metadata={
                    'case_name': case_name,
                    'paragraph_range': para_range,
                    'section_type': section['type']
                }
            ))
        
        return chunks
    
    def chunk_document(self, text: str, case_name: str = "Unknown Case") -> List[LegalChunk]:
        """
        Main chunking method: converts legal document to structured chunks.
        
        Args:
            text: Full document text
            case_name: Case identifier for metadata
            
        Returns:
            List of LegalChunk objects ready for embedding
        """
        # Step 1: Split into paragraphs
        paragraphs = self.split_into_paragraphs(text)
        
        # Step 2: Group by judicial sections
        sections = self.group_paragraphs_by_section(paragraphs)
        
        # Step 3: Create size-appropriate chunks from each section
        all_chunks = []
        for section in sections:
            chunks = self.create_chunks_from_section(section, case_name)
            all_chunks.extend(chunks)
        
        # Assign unique IDs
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_id = f"chunk_{i}_{chunk.chunk_type}"
        
        return all_chunks