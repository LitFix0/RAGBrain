"""
RAGBrain — Smart Extractor
Extracts project names directly from text using Python.
"""

import re
from typing import List, Dict


def extract_projects(chunks: List[Dict]) -> List[str]:
    sorted_chunks = sorted(chunks, key=lambda x: x.get("chunk_index", 0))
    full_text = "\n".join(c["text"] for c in sorted_chunks)

    # Debug: print what we're searching through
    print("[Extractor] Full text sample:")
    # Find and print the PROJECTS section area
    idx = full_text.upper().find("PROJECTS")
    if idx >= 0:
        print(full_text[idx:idx+500])
    else:
        print("PROJECTS section not found in text!")
        print("First 300 chars:", full_text[:300])

    # Find PROJECTS section
    match = re.search(
        r'PROJECTS\s*\n(.*?)(?=\n[A-Z][A-Z ]{2,}\n|\Z)',
        full_text,
        re.DOTALL | re.IGNORECASE
    )

    if not match:
        print("[Extractor] Could not find PROJECTS section with regex")
        return []

    projects_text = match.group(1)
    print("[Extractor] Projects section found:")
    print(projects_text[:400])

    project_names = []

    for line in projects_text.split('\n'):
        line = line.strip()
        # Match lines with | separator (project headers)
        if '|' in line and not line.startswith(('●', '•', '-', '*')) and len(line) > 3:
            name = line.split('|')[0].strip()
            if name and len(name) > 1:
                project_names.append(name)

    print(f"[Extractor] Found {len(project_names)} projects: {project_names}")
    return project_names


def is_listing_question(question: str) -> bool:
    keywords = ["list", "name", "what are", "how many", "mention",
                "all project", "3 project", "the project", "projects in",
                "projects mention", "show project"]
    return any(kw in question.lower() for kw in keywords)