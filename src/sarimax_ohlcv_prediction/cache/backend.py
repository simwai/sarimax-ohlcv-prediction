"""Cache backend using joblib.Memory with TTL support."""

import hashlib
import logging
import sys
import time
from pathlib import Path
from typing import Any

from joblib import Memory

from ..config import SETTINGS

logger = logging.getLogger(__name__)

_TTL_STORE_KEY = "__ttl_store__"


def _safe_key(key: str) -> str:
    """Convert cache key to a filesystem-safe string using SHA256 hash."""
    return hashlib.sha256(key.encode()).hexdigest()[:32]


class CacheManager:
    """Manages persistent cache with TTL-based invalidation using joblib.Memory."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        enabled: bool | None = None,
    ) -> None:
        """Initialize cache manager.

        Args:
            cache_dir: Directory for cache storage (default from SETTINGS)
            enabled: Whether caching is enabled (default from SETTINGS)
        """
        self._cache_dir = cache_dir or SETTINGS.cache_dir
        self._enabled = enabled if enabled is not None else SETTINGS.cache_enabled
        self._memory: Memory | None = None
        self._store_backend: Any = None
        self._ttl_store: dict[str, float] = {}  # key -> expiry timestamp
        self._init_memory()
        self._load_ttl_store()

    def _init_memory(self) -> None:
        """Initialize joblib Memory backend."""
        if self._enabled:
            try:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                self._memory = Memory(location=str(self._cache_dir), verbose=0)
                self._store_backend = self._memory.store_backend
                logger.debug("Cache initialized at %s", self._cache_dir)
            except Exception as e:
                logger.warning("Failed to initialize cache: %s. Caching disabled.", e)
                self._enabled = False
                self._memory = None
                self._store_backend = None
        else:
            self._memory = None
            self._store_backend = None

    @property
    def enabled(self) -> bool:
        """Check if cache is enabled and functional."""
        return self._enabled and self._memory is not None and self._store_backend is not None

    def _get_safe_key(self, key: str) -> str:
        """Get filesystem-safe key for joblib storage."""
        return _safe_key(key)

    def _ttl_store_key(self) -> str:
        """Get safe key for TTL store persistence."""
        return self._get_safe_key(_TTL_STORE_KEY)

    def _load_ttl_store(self) -> None:
        """Load TTL store from persistent storage."""
        if not self.enabled:
            return
        try:
            safe_key = self._ttl_store_key()
            if self._store_backend.contains_item(safe_key):
                self._ttl_store = self._store_backend.load_item(safe_key)
                logger.debug("Loaded TTL store with %d entries", len(self._ttl_store))
        except Exception as e:
            logger.debug("Failed to load TTL store: %s", e)
            self._ttl_store = {}

    def _save_ttl_store(self) -> None:
        """Save TTL store to persistent storage."""
        if not self.enabled:
            return
        try:
            safe_key = self._ttl_store_key()
            self._store_backend.dump_item(safe_key, self._ttl_store)
        except Exception as e:
            logger.debug("Failed to save TTL store: %s", e)

    def get(self, key: str) -> Any | None:
        """Retrieve value from cache if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if missing/expired/error
        """
        if not self.enabled:
            return None

        # Check TTL
        if key in self._ttl_store:
            if time.time() > self._ttl_store[key]:
                self.delete(key)
                return None
        else:
            # No TTL recorded - treat as expired
            return None

        try:
            safe_key = self._get_safe_key(key)
            if self._store_backend.contains_item(safe_key):
                return self._store_backend.load_item(safe_key)
        except Exception as e:
            logger.debug("Cache read failed for key %s: %s", key, e)
        return None

    def set(self, key: str, value: Any, ttl: int) -> None:
        """Store value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        if not self.enabled:
            return

        try:
            safe_key = self._get_safe_key(key)
            self._store_backend.dump_item(safe_key, value)
            self._ttl_store[key] = time.time() + ttl
            self._save_ttl_store()
            logger.debug("Cached key %s with TTL %ds", key, ttl)
        except Exception as e:
            logger.debug("Cache write failed for key %s: %s", key, e)

    def delete(self, key: str) -> bool:
        """Delete a key from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if key was deleted, False if not found
        """
        if not self.enabled:
            return False

        deleted = False
        try:
            safe_key = self._get_safe_key(key)
            if self._store_backend.contains_item(safe_key):
                self._store_backend.clear_item(safe_key)
                deleted = True
            if key in self._ttl_store:
                del self._ttl_store[key]
                deleted = True
            if deleted:
                self._save_ttl_store()
        except Exception as e:
            logger.debug("Cache delete failed for key %s: %s", key, e)
            return False

        if deleted:
            logger.debug("Deleted cache key %s", key)
        return deleted

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        if not self.enabled:
            return 0

        try:
            count = len(self._ttl_store)
            self._store_backend.clear()
            self._ttl_store.clear()
            self._save_ttl_store()
        except Exception:
            logger.exception("Cache clear failed")
            return 0
        else:
            logger.info("Cleared %d cache entries", count)
            return count

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache stats
        """
        if not self.enabled:
            return {"enabled": False, "entries": 0, "expired": 0, "size_bytes": 0}

        now = time.time()
        expired = sum(1 for exp in self._ttl_store.values() if now > exp)
        active = len(self._ttl_store) - expired

        # Estimate size
        size_bytes = 0
        try:
            for key in self._store_backend.get_items():
                try:
                    val = self._store_backend.load_item(key)
                    size_bytes += sys.getsizeof(val)
                except Exception:
                    pass
        except Exception:
            pass

        return {
            "enabled": True,
            "entries": len(self._ttl_store),
            "active": active,
            "expired": expired,
            "size_bytes": size_bytes,
            "cache_dir": str(self._cache_dir),
        }

    def inspect(self, limit: int = 50) -> list[dict[str, Any]]:
        """Inspect cache entries.

        Args:
            limit: Maximum entries to return

        Returns:
            List of entry info dicts
        """
        if not self.enabled:
            return []

        now = time.time()
        entries = []
        for key, expiry in list(self._ttl_store.items())[:limit]:
            ttl_remaining = max(0, int(expiry - now))
            entries.append({
                "key": key,
                "ttl_remaining": ttl_remaining,
                "expired": now > expiry,
            })
        return entries


# Global cache manager instance
_cache_manager: CacheManager | None = None


def get_cache_manager() -> CacheManager:
    """Get or create the global cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def reset_cache_manager() -> None:
    """Reset the global cache manager (for testing)."""
    global _cache_manager
    _cache_manager = None