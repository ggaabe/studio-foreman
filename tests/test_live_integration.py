"""Regressions from verifying the live fixes on the Studio (provenance, calibrated plans,
infrastructure failures that do not use caps).

  LI1  the mlx-serve adapter keeps the owner's default model even when that model is in the
       catalog: on the Studio, MLX-Serve.app's default is the catalog's fallback brain
       (Flash-Next), and a brain load by gpud must not leave the default on gpud's 27B
  LI2  a model the upgrade seeded as gpud's (the previous version recorded no server id) gets
       the id the server reports it under, and the adapter is told that copy is gpud's
"""
from __future__ import annotations

import os
import time

from foreman import studio
from foreman.supervisor import Supervisor

try:  # discovered with -s tests (tests/ on sys.path) or run from the project root
    from test_mlx_serve_adapter import (BRAIN, FALLBACK, FOREIGN, EngineCase, ManualClock, TmpCase, make_engine,
                                        make_fake, wait_for)
except ImportError:  # pragma: no cover
    from tests.test_mlx_serve_adapter import (BRAIN, FALLBACK, FOREIGN, EngineCase, ManualClock, TmpCase,
                                              make_engine, make_fake, wait_for)

FLASH_BASE = FALLBACK.rsplit("/", 1)[-1]  # how /api/ps names the copy MLX-Serve.app loaded by path


def defaults(fake) -> list:
    return [b for b in fake.bodies("/v1/load-model") if b.get("default")]


class LI1OwnersDefaultIsACatalogModel(EngineCase):

    def setUp(self):
        super().setUp()
        # The Studio: the owner's app loaded the fallback brain by path and it is the server's
        # (promoted) default; the user model the other tests use is not resident.
        self.fake.ready.pop(FOREIGN)
        self.fake.ready[FLASH_BASE] = int(75.3e9)
        self.fake.default_id, self.fake.default_promoted = FLASH_BASE, True

    def test_a_brain_load_by_gpud_does_not_keep_the_owners_default(self):
        self.load(BRAIN)
        self.assertEqual(self.fake.default_id, FLASH_BASE, "the owner's default stays the owner's")
        self.assertEqual(defaults(self.fake), [{"model": FLASH_BASE, "default": True}])
        self.assertIn(FLASH_BASE, self.fake.ready, "and the owner's model stays resident")
        self.assertFalse([b for b in self.fake.bodies("/v1/unload-model")])

    def test_gpuds_own_copy_as_the_default_is_not_restored(self):
        self.fake.ready.pop(FLASH_BASE)
        self.fake.default_id, self.fake.default_promoted = "", False
        self.load(BRAIN)                     # gpud's brain becomes the promoted default
        self.assertEqual(self.fake.default_id, BRAIN)
        self.engine.unload(BRAIN)
        wait_for(lambda: BRAIN not in self.engine.resident_models())
        self.fake.ready[BRAIN] = int(21e9)   # (stand-in for gpud's copy, as gpud loaded it)
        self.fake.default_id, self.fake.default_promoted = BRAIN, True
        self.engine.set_owned_id(BRAIN, BRAIN)
        self.load(FALLBACK)                  # gpud loads its fallback brain: default moves to it
        self.assertEqual(defaults(self.fake), [], "gpud's own previous default is not pinned back")


class LI2SeededOwnerGetsItsServerId(TmpCase):

    def make(self):
        catalog = studio.catalog()
        engine = make_engine(self.fake, self.tmp)
        sup = Supervisor(os.path.join(self.tmp, "foreman.db"), {"mlx-serve": engine}, self.clock,
                         studio.policy(idle_media_unload_s=None), catalog=catalog, pager=[].append)
        sup.recover()
        return engine, sup

    def run_until(self, sup, pred, timeout=20.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.clock.t += 1.0
            sup.tick()
            if pred():
                return True
            time.sleep(0.01)
        return False

    def test_the_upgrade_records_the_server_id_of_a_seeded_model(self):
        self.fake, self.clock = make_fake(), ManualClock()
        self.addCleanup(self.fake.stop)
        engine, sup = self.make()
        self.assertTrue(self.run_until(sup, lambda: BRAIN in sup._resident and sup._ours(BRAIN)))
        with sup.store.tx():  # the database as the version before provenance left it
            sup.store.conn.execute("DELETE FROM model_owners")
            sup.store.set_state(owners_init=0)
        sup.close()
        engine.close()
        engine, sup = self.make()
        self.addCleanup(engine.close)
        self.addCleanup(sup.close)
        self.assertEqual([e["data"]["owned"] for e in sup.store.events("ownership_seeded")], [[BRAIN]])
        self.assertEqual(sup.store.owners()[BRAIN]["server_id"], BRAIN)
        self.assertEqual(engine.owned_id(BRAIN), BRAIN, "the adapter knows which copy is gpud's")
        self.assertTrue(sup._ours(BRAIN))
        self.assertFalse(sup._ours(FOREIGN))


if __name__ == "__main__":
    import unittest
    unittest.main()
