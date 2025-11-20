"""
Unit tests for gconvex package with timing support.

This module provides a custom TestCase base class that automatically logs execution time 
for each test, making it easy to identify performance bottlenecks in the test suite.

All test classes should inherit from TimedTestCase instead of unittest.TestCase.
"""
import unittest
import time
import sys
import atexit
import signal


class TimedTestCase(unittest.TestCase):
    """Base test case that automatically tracks and logs execution time.
    
    Optional timeout support (Unix only):
    - Set max_test_time (in seconds) to enforce a timeout on each test
    - Set to None (default) to disable timeouts
    - Example: max_test_time = 300  # Fail tests taking >5 minutes
    """
    
    # Class-level storage for all test times across all test cases
    _all_test_times = []
    _tests_run = 0
    _summary_registered = False
    
    # Optional timeout (set to None to disable, or number of seconds to enforce)
    max_test_time = 10  # Default: 10 seconds per test (set to None to disable)
    
    def setUp(self):
        """Start timing before each test."""
        self._test_start_time = time.time()
        
        # Set up timeout if configured (Unix only)
        if self.max_test_time is not None and hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(int(self.max_test_time))
        
        super().setUp()
    
    def tearDown(self):
        """Record timing after each test."""
        # Cancel timeout if it was set
        if self.max_test_time is not None and hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        
        super().tearDown()
        if hasattr(self, '_test_start_time'):
            elapsed = time.time() - self._test_start_time
            test_name = f"{self.__class__.__name__}.{self._testMethodName}"
            TimedTestCase._all_test_times.append((test_name, elapsed))
            TimedTestCase._tests_run += 1
            
            # Print timing for slow tests (>1s) in verbose mode
            if elapsed > 1.0 and hasattr(sys.stdout, 'write'):
                sys.stdout.write(f" ({elapsed:.2f}s)")
                sys.stdout.flush()
    
    def _timeout_handler(self, signum, frame):
        """Handle test timeout."""
        test_name = f"{self.__class__.__name__}.{self._testMethodName}"
        self.fail(f"Test timed out after {self.max_test_time} seconds")
    
    @classmethod
    def setUpClass(cls):
        """Called before tests in the class are run."""
        super().setUpClass()
        
        # Register the summary printer once
        if not TimedTestCase._summary_registered:
            TimedTestCase._summary_registered = True
            atexit.register(print_timing_summary)


def print_timing_summary():
    """Print timing summary at the end of all tests."""
    if not TimedTestCase._all_test_times:
        return
    
    print("\n" + "=" * 70)
    print("Test Timing Summary")
    print("=" * 70)
    
    total_time = sum(t for _, t in TimedTestCase._all_test_times)
    print(f"Total test time: {total_time:.2f}s ({len(TimedTestCase._all_test_times)} tests)")
    print()
    
    # Sort by time and show slowest tests
    sorted_times = sorted(TimedTestCase._all_test_times, key=lambda x: x[1], reverse=True)
    
    # Show top 20 slowest tests
    print("Top 20 slowest tests:")
    for test_name, elapsed in sorted_times[:20]:
        print(f"  {elapsed:6.2f}s - {test_name}")
    
    # Show count of tests taking more than 5 seconds
    slow_tests = [t for _, t in sorted_times if t > 5.0]
    if len(slow_tests) > 20:
        print()
        print(f"Additional {len(slow_tests) - 20} tests taking >5s")
    print("=" * 70)
