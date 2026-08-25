import asyncio
import itertools
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


DEFAULT_MAX_CONCURRENT = int(os.getenv("KEY_ROTATOR_MAX_CONCURRENT", "50"))


class NoAvailableAPIKeysError(RuntimeError):
    """No configured API keys remain available in the current process."""


class KeyRotator:
    """Round-robin API key rotator with concurrency control."""

    def __init__(
        self, keys: list[str], max_concurrent: int = DEFAULT_MAX_CONCURRENT
    ) -> None:
        if not keys:
            raise ValueError("At least one key must be provided")
        # De-duplicate keys without changing their configured order.
        self._keys = list(dict.fromkeys(keys))
        self._cycle = itertools.cycle(self._keys)
        self._disabled_keys: set[str] = set()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._state_lock = asyncio.Lock()

    @classmethod
    def from_env(
        cls, env_var: str, max_concurrent: int = DEFAULT_MAX_CONCURRENT
    ) -> "KeyRotator":
        raw = os.getenv(env_var)
        if not raw:
            raise ValueError(f"{env_var} is not set")
        keys = [k.strip() for k in raw.split(";") if k.strip()]
        if not keys:
            raise ValueError(f"{env_var} is empty after parsing")
        return cls(keys, max_concurrent)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[str]:
        """Acquire a slot and yield the next active key in round-robin order."""
        async with self._semaphore:
            async with self._state_lock:
                key = None
                for _ in range(len(self._keys)):
                    candidate = next(self._cycle)
                    if candidate not in self._disabled_keys:
                        key = candidate
                        break
                if key is None:
                    raise NoAvailableAPIKeysError(
                        "All configured API keys are unavailable in this process"
                    )
            yield key

    async def disable(self, key: str) -> bool:
        """Disable a failed key for the rest of the current process.

        Returns True only when this call changed the key's state.
        """
        async with self._state_lock:
            if key not in self._keys or key in self._disabled_keys:
                return False
            self._disabled_keys.add(key)
            return True

    @property
    def key_count(self) -> int:
        return len(self._keys)

    @property
    def active_key_count(self) -> int:
        return len(self._keys) - len(self._disabled_keys)


_rotators: dict[str, KeyRotator] = {}


def get_rotator(
    env_var: str, max_concurrent: int = DEFAULT_MAX_CONCURRENT
) -> KeyRotator:
    """Returns a shared KeyRotator per env var name, creating on first access."""
    if env_var not in _rotators:
        _rotators[env_var] = KeyRotator.from_env(env_var, max_concurrent)
    return _rotators[env_var]
