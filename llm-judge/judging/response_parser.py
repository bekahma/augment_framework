"""
Response parsing from LLM, extract text for spreadsheet.
"""

import re
from typing import Tuple


class ResponseParser:
    """Parses LLM responses for judgments."""
    
    @staticmethod
    def parse_simple(content: str) -> str:
        """Parse simple TRUE/FALSE response."""
        verdict = content.strip().upper()
        return verdict if verdict in ["TRUE", "FALSE"] else "UNKNOWN"
    
    @staticmethod
    def parse_with_reasoning(content: str) -> Tuple[str, str]:
        """
        Parse response with reasoning.
        
        Returns:
            Tuple of (decision, reason)
        """
        # Parse decision
        m_dec = re.search(r'\bDECISION:\s*(TRUE|FALSE)\b', content, re.IGNORECASE)
        decision = m_dec.group(1).upper() if m_dec else (
            content.upper() if content.upper() in {"TRUE", "FALSE"} else "UNKNOWN"
        )
        
        # Parse reasoning
        m_rea = re.search(r'\bREASON:\s*(.*)', content, re.IGNORECASE | re.DOTALL)
        reason = m_rea.group(1).strip() if m_rea else (
            content if decision == "UNKNOWN" else ""
        )
        
        return decision, reason