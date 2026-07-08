r"""Seamless Interaction identifier parsing.

The old working downloader used a regex like
`V\d+_S\d+_I\d+_P[0-9A-Za-z]+`. This module keeps that behavior but returns a
structured result and supports alphanumeric participant IDs such as `P0844A`.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

FILE_ID_PATTERN = re.compile(
    r"(?P<vendor>V\d+)_S(?P<session>\d+)_I(?P<interaction>\d+)_P(?P<participant>[0-9A-Za-z]+)"
)


@dataclass(frozen=True)
class ParsedFileId:
    """Structured Seamless file identifier."""

    file_id: str
    interaction_key: str
    participant_id: str
    vendor: str
    session: str
    interaction: str


def parse_file_id(file_id: str) -> ParsedFileId:
    """Parse a Seamless file ID.

    Raises
    ------
    ValueError
        If the file ID does not match the expected Seamless pattern.
    """
    text = str(file_id)
    m = FILE_ID_PATTERN.search(text)
    if not m:
        raise ValueError(f"Could not parse Seamless file_id: {file_id}")
    vendor = m.group("vendor")
    session = m.group("session")
    interaction = m.group("interaction")
    participant = "P" + m.group("participant")
    interaction_key = f"{vendor}_S{session}_I{interaction}"
    return ParsedFileId(
        file_id=m.group(0),
        interaction_key=interaction_key,
        participant_id=participant,
        vendor=vendor,
        session=session,
        interaction=interaction,
    )


def interaction_key_from_file_id(file_id: str) -> str:
    """Return the interaction key for a Seamless file ID."""
    return parse_file_id(file_id).interaction_key
