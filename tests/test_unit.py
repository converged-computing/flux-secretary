"""Unit tests that need no Flux
Only the launch ladder is pure logic. Everything else talks to a broker and is
covered by test_integration.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fluxsecretary.launch import ladder


def test_ladder_orders_most_to_least_specific():
    plans = ladder({"nodes": 5, "cores": 15}, 5)
    assert [p.tasks for p in plans][:3] == [15, 5, None]
    print("OK ladder is ordered")


def test_ladder_never_exceeds_the_allocation():
    assert ladder({"nodes": 5, "cores": 20}, 10)[0].nodes == 5
    print("OK ladder caps at the allocation")


if __name__ == "__main__":
    test_ladder_orders_most_to_least_specific()
    test_ladder_never_exceeds_the_allocation()
    print("\nunit tests passed")
