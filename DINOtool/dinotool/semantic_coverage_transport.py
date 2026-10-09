"""Uniform single-JSON-fence transport compatibility, without schema salvage."""
import re

from .semantic_coverage_input import parse_reply


IMPLEMENTATION='fixed20-semantic-coverage-single-json-fence-transport-v1-20261009'


def parse_transport(reply,record):
    value=reply.strip()
    if value.startswith('```'):
        match=re.fullmatch(r'```json\s*\n([\s\S]*?)\n```',value)
        if match is None:raise ValueError('Reply must be exactly one complete json fence.')
        value=match.group(1)
    return parse_reply(value,record)
