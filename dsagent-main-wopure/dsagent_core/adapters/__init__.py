"""Adapters Module - Connect DSAgent Core to various agent frameworks"""

from dsagent_core.adapters.metagpt_adapter import MetaGPTAdapter

from dsagent_core.adapters.autogen_adapter import (
    AutoGenAdapter,
    AUTOGEN_AVAILABLE,
    AUTOGEN_VERSION,
    create_dsagent_autogen_adapter,
)

try:
    from dsagent_core.adapters.metagpt_lats_adapter import MetaGPTLATSAdapter
    METAGPT_LATS_AVAILABLE = True
except ImportError:
    MetaGPTLATSAdapter = None
    METAGPT_LATS_AVAILABLE = False

try:
    from dsagent_core.adapters.autogen_lats_adapter import (
        AutoGenLATSAdapter,
        create_autogen_lats,
    )
    AUTOGEN_LATS_AVAILABLE = True
except ImportError:
    AutoGenLATSAdapter = None
    create_autogen_lats = None
    AUTOGEN_LATS_AVAILABLE = False

__all__ = [
    "MetaGPTAdapter",
    "AutoGenAdapter",
    "AUTOGEN_AVAILABLE",
    "AUTOGEN_VERSION",
    "create_dsagent_autogen_adapter",
    "MetaGPTLATSAdapter",
    "METAGPT_LATS_AVAILABLE",
    "AutoGenLATSAdapter",
    "create_autogen_lats",
    "AUTOGEN_LATS_AVAILABLE",
]
