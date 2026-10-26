"""Clocks.

gpud measures leases, load timeouts, cancel grace, cap accounting and SLAs on *supervisor time*:
a monotonic timeline persisted in SQLite. Within one process it advances with a monotonic source
when the clock object offers one (`clock.monotonic()`), so NTP steps, RTC corrections and manual
date changes cannot kill a healthy job or blind the dead-man switch. Across a process restart it
is re-baselined from the persisted (time, wall, monotonic) triple: by the monotonic delta when the
source is still the same timeline (same boot), otherwise by the wall-clock delta (never negative).
Without a monotonic source a wall-clock jump larger than Policy.max_clock_step_s inside one
process is treated as a step and ignored (recorded as a clock_step event).
"""
from __future__ import annotations

import time


class SystemClock:
    """The production clock: wall time for humans, time.monotonic() for everything gpud times."""

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()
