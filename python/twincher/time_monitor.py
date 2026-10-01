# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock


@dataclass
class _FunctionStats:
    """Internal statistics for one monitored function/block."""
    running: bool = False
    start_time: float | None = None

    # Number of calls represented by the current start()/stop() interval.
    current_expected_calls: int = 1

    calls_count: int = 0
    elapsed_time: float = 0.0
    shortest_time: float | None = None
    longest_time: float | None = None
    first_call_time = None
    first_call_count = None


def _synchronize_cuda():
    """Wait for pending GPU work, if PyTorch uses CUDA (for timing of asynchronous operations)."""
    import torch
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


class TimeMonitor:
    """
    Monitor elapsed time spent in named functions/blocks.

    Examples
    --------
    Single block:

        monitor.start("load_data")
        load_data()
        monitor.stop("load_data")

    Multiple calls:

        monitor.start("foo", 100)
        for _ in range(100):
            foo()
        monitor.stop("foo")

    The second example records 100 calls and uses the total elapsed
    time divided by 100 for the average time per call.
    """

    def __init__(self, file_name: str | None, disable: bool = False):
        # file_name can be None only for a disabled monitor, for which no report is saved
        if file_name is None and not disable:
            raise ValueError("file_name is required for an enabled TimeMonitor")
        self.file_name = Path(file_name) if file_name is not None else None
        self.disable = disable

        self._init_time = time.perf_counter()

        # Names are kept in insertion order.
        self.function_names: list[str] = []

        # Statistics indexed by function name.
        self._stats: dict[str, _FunctionStats] = {}

        self._lock = RLock()
        self._report_saved = False

    def start(
        self,
        function_name: str,
        expected_calls: int = 1,
        torch_sync: bool = False,
    ) -> None:
        """
        Start timing a function/block.

        Parameters
        ----------
        function_name:
            Name of the function/block being measured.

        expected_calls:
            Number of function calls represented by this timing interval.

            For example:

                monitor.start("foo", 100)
                for _ in range(100):
                    foo()
                monitor.stop("foo")

            records 100 calls.

        Raises
        ------
        TypeError
            If function_name is not a string or expected_calls is not an int.

        ValueError
            If expected_calls is less than 1, or the function is already
            running.
        """
        if self.disable:
            return

        if not isinstance(function_name, str):
            raise TypeError("function_name must be a string")

        if not function_name:
            raise ValueError("function_name cannot be empty")

        if not isinstance(expected_calls, int):
            raise TypeError("expected_calls must be an integer")

        if expected_calls < 1:
            raise ValueError("expected_calls must be at least 1")

        with self._lock:
            if function_name not in self._stats:
                self.function_names.append(function_name)
                self._stats[function_name] = _FunctionStats()

            stats = self._stats[function_name]

            if stats.running:
                raise ValueError(
                    f"Function '{function_name}' has already been started "
                    "and has not been stopped."
                )

            if torch_sync:
                _synchronize_cuda()

            stats.running = True
            stats.start_time = time.perf_counter()
            stats.current_expected_calls = expected_calls

    def stop(self, function_name: str, torch_sync = False) -> None:
        """
        Stop timing a function/block and update its statistics.

        The number of calls recorded is the expected_calls value supplied
        to the corresponding start().

        Raises
        ------
        TypeError
            If function_name is not a string.

        ValueError
            If function_name was never started or is not currently running.
        """
        if self.disable:
            return

        if not isinstance(function_name, str):
            raise TypeError("function_name must be a string")

        with self._lock:
            if function_name not in self._stats:
                raise ValueError(
                    f"Function '{function_name}' has never been started."
                )

            stats = self._stats[function_name]

            if not stats.running or stats.start_time is None:
                raise ValueError(
                    f"Function '{function_name}' is not currently running."
                )

            if torch_sync:
                _synchronize_cuda()

            elapsed = time.perf_counter() - stats.start_time
            call_count = stats.current_expected_calls

            stats.running = False
            stats.start_time = None

            # Accumulate the number of calls represented by this interval.
            stats.calls_count += call_count

            # Accumulate total elapsed time.
            stats.elapsed_time += elapsed

            if stats.first_call_time is None:
                stats.first_call_time = elapsed
                stats.first_call_count = call_count
                return

            # For shortest/longest, we want the time per individual call,
            # not the total time for the whole batch.
            average_call_time = elapsed / call_count

            if (
                stats.shortest_time is None
                or average_call_time < stats.shortest_time
            ):
                stats.shortest_time = average_call_time

            if (
                stats.longest_time is None
                or average_call_time > stats.longest_time
            ):
                stats.longest_time = average_call_time

    def save_report(self) -> None:
        if self.file_name is None:
            return

        with self._lock:
            report_time = time.perf_counter()
            monitoring_time = report_time - self._init_time

            self.file_name.parent.mkdir(parents=True, exist_ok=True)

            with self.file_name.open("w", encoding="utf-8") as file:
                file.write(
                    "Time monitoring report (time is given in seconds)\n"
                    f"{'=' * 90}\n"
                    f"Monitoring time: {monitoring_time:.6f} s\n"
                    "\n"
                )
                if self.disable:
                    file.write("Time monitor has been disabled\n")
                    self._report_saved = True
                    return    

                headers = (
                    "Function",
                    "Calls",
                    "Total",
                    "Shortest",
                    "Longest",
                    "Average",
                    "% of time",
                )

                file.write(
                    f"{headers[0]:<24}"
                    f"{headers[1]:>11}"
                    f"{headers[2]:>11}"
                    f"{headers[3]:>11}"
                    f"{headers[4]:>11}"
                    f"{headers[5]:>11}"
                    f"{headers[6]:>11}\n"
                )

                file.write("-" * 90 + "\n")

                for function_name in self.function_names:
                    stats = self._stats[function_name]

                    if monitoring_time > 0:
                        percentage = (
                            stats.elapsed_time
                            / monitoring_time
                            * 100.0
                        )
                    else:
                        percentage = 0.0

                    if stats.shortest_time is not None:
                        shortest = f"{stats.shortest_time:8.2e}"
                    else:
                        shortest = "-"

                    if stats.longest_time is not None:
                        longest = f"{stats.longest_time:8.2e}"
                    else:
                        longest = "-"

                    if stats.calls_count - stats.first_call_count > 0:
                        average = (
                            (stats.elapsed_time - stats.first_call_time) / (stats.calls_count - stats.first_call_count)
                        )
                        average_string = f"{average:8.2e}"
                    else:
                        average_string = "-"

                    file.write(
                        f"{function_name:<24.24}"
                        f"{stats.calls_count:>11.2e}"
                        f"{stats.elapsed_time:>11.2e}"
                        f"{shortest:>11}"
                        f"{longest:>11}"
                        f"{average_string:>11}"
                        f"{percentage:>10.2f}%\n"
                    )

                file.write("-" * 90 + "\n")

            self._report_saved = True

    def __del__(self):
        """Save the report automatically as a fallback."""
        try:
            if not self._report_saved:
                self.save_report()
        except Exception:
            # Never let errors escape from __del__.
            pass

