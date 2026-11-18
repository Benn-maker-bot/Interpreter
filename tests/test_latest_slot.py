"""Tests for LatestSlot, the one-item mailbox for live previews.

Electronics analogy: LatestSlot is a single latch register, not a FIFO. A new
write overwrites the old value; a read clears it. `overwritten` counts the
values that were written but never read (dropped previews).
"""
import threading

from interpreter import LatestSlot


def test_new_slot_is_empty():
    assert LatestSlot().take() is None


def test_new_slot_has_dropped_nothing():
    assert LatestSlot().overwritten == 0


def test_take_returns_what_was_put():
    slot = LatestSlot()
    slot.put("a")
    assert slot.take() == "a"


def test_take_empties_the_slot():
    slot = LatestSlot()
    slot.put("a")
    slot.take()
    assert slot.take() is None


def test_put_overwrites_and_counts_the_dropped_item():
    # Three puts with no take: only the newest survives, two are dropped.
    slot = LatestSlot()
    slot.put("a")
    slot.put("b")
    slot.put("c")
    assert slot.take() == "c"
    assert slot.overwritten == 2


def test_put_after_take_is_not_counted_as_overwrite():
    # The first item was read, so replacing it drops nothing.
    slot = LatestSlot()
    slot.put("a")
    slot.take()
    slot.put("b")
    assert slot.overwritten == 0
    assert slot.take() == "b"


def test_slot_holds_any_object():
    # The capture thread puts audio arrays and tuples, not just strings.
    slot = LatestSlot()
    item = ("mic", b"\x00\x01", 1.5)
    slot.put(item)
    assert slot.take() is item


def test_no_item_is_lost_or_duplicated_between_threads():
    # One thread puts 20,000 numbered items while another keeps taking.
    # Every item must end up either taken once or counted as overwritten:
    #     taken + overwritten == put
    # This is what the lock guarantees. (Without it the test would usually
    # still pass in CPython, so it proves the bookkeeping, not the lock.)
    slot = LatestSlot()
    count = 20_000
    taken = []
    done = threading.Event()

    def producer():
        for i in range(count):
            slot.put(i)
        done.set()

    def consumer():
        while not done.is_set():
            item = slot.take()
            if item is not None:
                taken.append(item)

    threads = [threading.Thread(target=producer), threading.Thread(target=consumer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    last = slot.take()             # anything left after the consumer stopped
    if last is not None:
        taken.append(last)

    assert len(taken) + slot.overwritten == count
    assert len(set(taken)) == len(taken)       # nothing taken twice
    assert taken == sorted(taken)              # newest wins, so order is kept
    assert taken[-1] == count - 1              # the very last put is never lost
