"""Foreman: a GPU-residency supervisor that always brings the deciding brain back."""

import logging

from .policy import CATALOG, ModelSpec, Policy
from .supervisor import AdmissionError, Supervisor

__all__ = ["CATALOG", "ModelSpec", "Policy", "AdmissionError", "Supervisor"]

# Library convention: the gpud daemon configures logging; importing foreman stays silent.
logging.getLogger("foreman").addHandler(logging.NullHandler())
