# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for ``transformers.pytorch_utils`` helpers.

The most important test here is :class:`CompileCompatibleMethodLruCacheTest.
test_module_instance_is_garbage_collected` -- a direct regression test for
https://github.com/huggingface/transformers/issues/45412 ("RT-DETR models do
not release memory when deleted / garbage-collected"). The root cause of that
issue was a class-level ``functools.lru_cache`` holding a strong reference to
every instance via the cached ``self`` argument; this test asserts that a
``weakref`` to an instance resolves to ``None`` after the instance is deleted
and ``gc.collect()`` runs, which is only possible if the fix in
``pytorch_utils.py`` keeps the cache per-instance.
"""

from __future__ import annotations

import gc
import unittest
import weakref

import torch
from torch import nn

from transformers.pytorch_utils import compile_compatible_method_lru_cache


class _LeakCanary(nn.Module):
    """Minimal ``nn.Module`` with a cached method.

    The cached method returns a fresh tensor on each miss, and records hit/miss
    counts on the instance so tests can assert that the cache is actually used.
    """

    def __init__(self, value: float = 1.0):
        super().__init__()
        self.value = value
        self.calls = 0

    @compile_compatible_method_lru_cache(maxsize=8)
    def cached(self, width: int, height: int) -> torch.Tensor:
        self.calls += 1
        return torch.full((height, width), self.value)


# A module-level function decorated with the same helper, mirroring the real
# ``dinov3_vit.get_patches_center_coordinates`` usage pattern (no ``self``).
_module_level_calls = {"n": 0}


@compile_compatible_method_lru_cache(maxsize=8)
def _module_level_cached(n: int, m: int) -> torch.Tensor:
    _module_level_calls["n"] += 1
    return torch.arange(n * m).reshape(n, m)


class CompileCompatibleMethodLruCacheTest(unittest.TestCase):
    # ---------------- behavioural parity ----------------

    def test_module_method_caches_identical_calls(self):
        m = _LeakCanary()
        t1 = m.cached(4, 2)
        t2 = m.cached(4, 2)
        # Cache hit: same tensor object returned twice, call count still 1.
        self.assertIs(t1, t2)
        self.assertEqual(m.calls, 1)

        # Miss on different args increments the call count.
        _ = m.cached(3, 2)
        self.assertEqual(m.calls, 2)

    def test_distinct_instances_have_independent_caches(self):
        a = _LeakCanary(value=1.0)
        b = _LeakCanary(value=2.0)

        ta = a.cached(2, 2)
        tb = b.cached(2, 2)

        # Per-instance caches -> values track the instance they came from.
        self.assertTrue(torch.equal(ta, torch.full((2, 2), 1.0)))
        self.assertTrue(torch.equal(tb, torch.full((2, 2), 2.0)))
        self.assertEqual(a.calls, 1)
        self.assertEqual(b.calls, 1)

    def test_module_level_function_still_cached(self):
        # Reset shared counter.
        _module_level_calls["n"] = 0

        t1 = _module_level_cached(3, 4)
        t2 = _module_level_cached(3, 4)
        self.assertIs(t1, t2)
        self.assertEqual(_module_level_calls["n"], 1)

        _ = _module_level_cached(5, 2)
        self.assertEqual(_module_level_calls["n"], 2)

    # ---------------- the regression test ----------------

    def test_module_instance_is_garbage_collected(self):
        """Regression test for issue #45412.

        Before the fix, the class-level ``lru_cache`` kept ``self`` alive
        forever via its internal key cache, so ``weakref.ref(instance)``
        resolved to something non-``None`` even after ``del instance`` and
        ``gc.collect()``. After the fix, the cache lives on the instance, so
        deleting the instance releases the cache and the weakref resolves to
        ``None``.
        """
        instance = _LeakCanary()

        # Populate the cache so the leak-vulnerable code path is exercised.
        _ = instance.cached(2, 2)
        _ = instance.cached(4, 4)
        _ = instance.cached(8, 8)

        ref = weakref.ref(instance)
        self.assertIsNotNone(ref())

        del instance
        gc.collect()

        self.assertIsNone(
            ref(),
            "Instance was not garbage-collected -- the lru_cache is still "
            "holding a reference to `self`. See "
            "https://github.com/huggingface/transformers/issues/45412",
        )

    def test_tensor_is_freed_after_instance_delete(self):
        """The cached tensor itself must also be released with the instance."""
        instance = _LeakCanary()
        cached_tensor = instance.cached(2, 2)
        tensor_ref = weakref.ref(cached_tensor)

        # Drop both our explicit reference and the instance that owns the cache.
        del cached_tensor
        del instance
        gc.collect()

        self.assertIsNone(
            tensor_ref(),
            "Cached tensor survived instance deletion -- the cache leaked.",
        )

    # ---------------- non-Module first arg is unchanged ----------------

    def test_non_module_first_arg_uses_class_level_cache(self):
        """A plain function (int as first arg) must not trip the module-only
        code path. This guards against the regression described in the prior
        closed PR (``cannot create weak reference to 'int' object``).
        """
        _module_level_calls["n"] = 0
        self.assertIsNotNone(_module_level_cached(2, 3))
        # Should simply work -- no exception about weakrefs or ids.


if __name__ == "__main__":
    unittest.main()
