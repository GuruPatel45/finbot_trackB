"""
backend/database/cache.py
==========================
Redis-based async cache with in-memory fallback.
Used for market data, prices, news sentiment caching.

Strategy:
  - Try Redis first (fast, shared across instances)
  - Fall back to in-memory TTL cache if Redis unavailable
  - All cache keys are namespaced: finbot:{data_type}:{key}
"""

import json
import logging
from datetime import timedelta
from typing import Optional, Any
from cachetools import TTLCache

logger = logging.getLogger(__name__)

# In-memory fallback cache
_fallback_cache = TTLCache(maxsize=1000, ttl=300)
_redis_available = False
_redis_client = None


async def init_redis(redis_url: str) -> bool:
    """Initialize Redis connection. Returns True if successful."""
    global _redis_client, _redis_available
    try:
        import redis.asyncio as aioredis
        _redis_client = aioredis.from_url(redis_url, decode_responses=True, socket_timeout=3)
        await _redis_client.ping()
        _redis_available = True
        logger.info("✅ Redis connected successfully")
        return True
    except Exception as e:
        logger.warning(f"⚠️  Redis unavailable ({e}). Using in-memory cache fallback.")
        _redis_available = False
        return False


async def cache_get(key: str) -> Optional[Any]:
    """Get value from cache (Redis → in-memory fallback)."""
    ns_key = f"finbot:{key}"

    if _redis_available and _redis_client:
        try:
            val = await _redis_client.get(ns_key)
            if val:
                return json.loads(val)
        except Exception as e:
            logger.warning(f"Redis GET failed: {e}")

    # Fallback to in-memory
    return _fallback_cache.get(ns_key)


async def cache_set(key: str, value: Any, ttl: int = 300) -> None:
    """Set value in cache with TTL (seconds)."""
    ns_key = f"finbot:{key}"
    serialized = json.dumps(value)

    if _redis_available and _redis_client:
        try:
            await _redis_client.setex(ns_key, timedelta(seconds=ttl), serialized)
            return
        except Exception as e:
            logger.warning(f"Redis SET failed: {e}")

    # Fallback to in-memory (TTLCache handles expiry automatically)
    _fallback_cache[ns_key] = value


async def cache_delete(key: str) -> None:
    """Delete a cache key."""
    ns_key = f"finbot:{key}"
    if _redis_available and _redis_client:
        try:
            await _redis_client.delete(ns_key)
        except Exception:
            pass
    _fallback_cache.pop(ns_key, None)


async def cache_flush_pattern(pattern: str) -> int:
    """Flush all keys matching pattern (Redis only)."""
    if _redis_available and _redis_client:
        try:
            keys = await _redis_client.keys(f"finbot:{pattern}*")
            if keys:
                return await _redis_client.delete(*keys)
        except Exception as e:
            logger.warning(f"Redis FLUSH failed: {e}")
    return 0


def get_cache_status() -> dict:
    """Return cache health status for monitoring."""
    return {
        "redis_available": _redis_available,
        "fallback_cache_size": len(_fallback_cache),
        "fallback_cache_maxsize": _fallback_cache.maxsize,
    }
