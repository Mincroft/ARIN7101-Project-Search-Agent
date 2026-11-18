"""Run independent read-only SQLite searches outside the sampling event loop.

Workers call the original retrieval implementation. Per-episode tool and
curation state remains in the original environment in the parent process.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

from sec_rl import environment
from sec_rl.retrieval import CorpusIndex, get_index


def _search(data_dir, method, arguments):
    return getattr(get_index(data_dir), method)(**arguments)


class ProcessCorpusIndex(CorpusIndex):
    def __init__(self, data_dir, executor):
        super().__init__(Path(data_dir))
        self.executor = executor

    def bm25_search(self, query, *, k, snippet_chars):
        return self.executor.submit(
            _search,
            str(self.data_dir),
            "bm25_search",
            {"query": query, "k": k, "snippet_chars": snippet_chars},
        ).result()

    def grep(self, pattern, *, k, snippet_chars, case_sensitive=False):
        return self.executor.submit(
            _search,
            str(self.data_dir),
            "grep",
            {
                "pattern": pattern,
                "k": k,
                "snippet_chars": snippet_chars,
                "case_sensitive": case_sensitive,
            },
        ).result()


class RetrievalPool:
    def __init__(self, workers=8):
        self.executor = ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        )
        self.previous_factory = None

    def __enter__(self):
        self.previous_factory = environment.get_index

        @lru_cache(maxsize=4)
        def factory(data_dir):
            return ProcessCorpusIndex(data_dir, self.executor)

        environment.get_index = factory
        return self

    def __exit__(self, *_):
        environment.get_index = self.previous_factory
        self.executor.shutdown(wait=True, cancel_futures=True)
