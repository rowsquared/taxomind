"""Dramatiq broker configuration.

Reads ``REDIS_URL`` from environment and configures a ``RedisBroker``.
This module **must** be imported before any ``@dramatiq.actor`` decorators
are evaluated.
"""

from __future__ import annotations

import os

import dramatiq
from dramatiq.brokers.redis import RedisBroker

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

redis_broker = RedisBroker(url=REDIS_URL)
dramatiq.set_broker(redis_broker)
