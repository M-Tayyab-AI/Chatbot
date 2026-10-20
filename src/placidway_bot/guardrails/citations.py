"""Citation guardrail: a grounded answer must cite at least one of the sources it was given, and nothing else."""
from __future__ import annotations


def citation_issues(cited: list[str], provided: list[str], answer_found: bool) -> list[str]:
    issues = []
    unknown = [c for c in cited if c not in provided]
    if unknown:
        issues.append(f"You cited sources that do not exist: {', '.join(unknown)}. Only cite {', '.join(provided)}.")
    if answer_found and not [c for c in cited if c in provided]:
        issues.append("You gave an answer without citing any source. Cite the source ids you used.")
    return issues
