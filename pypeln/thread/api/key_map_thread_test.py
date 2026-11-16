import threading
import time
import typing as tp

import hypothesis as hp
from hypothesis import strategies as st

import pypeln as pl

MAX_EXAMPLES = 15
T = tp.TypeVar("T")


def _key(x):
    return x % 5


# ----------------------------------------------------------------
# correctness
# ----------------------------------------------------------------


@hp.given(nums=st.lists(st.integers()))
@hp.settings(max_examples=MAX_EXAMPLES)
def test_key_map_identity(nums: tp.List[int]):
    nums_pl = list(pl.thread.key_map(lambda x: x, _key, nums, workers=4))

    assert sorted(nums_pl) == sorted(nums)


@hp.given(nums=st.lists(st.integers()))
@hp.settings(max_examples=MAX_EXAMPLES)
def test_key_map_square(nums: tp.List[int]):
    nums_py = [x**2 for x in nums]

    nums_pl = list(pl.thread.key_map(lambda x: x**2, _key, nums, workers=3))

    assert sorted(nums_pl) == sorted(nums_py)


@hp.given(nums=st.lists(st.integers()))
@hp.settings(max_examples=MAX_EXAMPLES)
def test_key_map_pipe(nums: tp.List[int]):
    nums_pl = nums | pl.thread.key_map(lambda x: x, _key, workers=3) | list

    assert sorted(nums_pl) == sorted(nums)


@hp.given(nums=st.lists(st.integers()))
@hp.settings(max_examples=MAX_EXAMPLES)
def test_key_map_partial(nums: tp.List[int]):
    f = pl.thread.key_map(lambda x: x + 1, _key, workers=3)
    nums_pl = list(f(nums))

    assert sorted(nums_pl) == sorted(x + 1 for x in nums)


def test_single_worker_preserves_order():
    # with one partition everything is serial, so input order is kept
    data = list(range(50))

    result = list(
        pl.thread.key_map(
            lambda x: x + 1,
            key=lambda x: x % 3,
            stage=data,
            workers=1,
        )
    )

    assert result == [x + 1 for x in data]


def test_key_hashable_with_none_and_tuple_keys():
    data = [None, (1, 2), None, (1, 2), "a", None]

    def f(x):
        return x

    result = list(pl.thread.key_map(f, key=lambda x: x, stage=data, workers=3))

    assert sorted(result, key=str) == sorted(data, key=str)


def test_key_map_empty():
    # every partition is empty and the stage must still finish cleanly,
    # running on_start/on_done once for every partition
    n_workers = 4
    started = []
    done = []
    lock = threading.Lock()

    def on_start(worker_info):
        with lock:
            started.append(worker_info.index)
        return {}

    def on_done(stage_status):
        with lock:
            done.append(stage_status.done)

    result = list(
        pl.thread.key_map(
            lambda x: x,
            _key,
            [],
            workers=n_workers,
            on_start=on_start,
            on_done=on_done,
        )
    )

    assert result == []
    assert sorted(started) == list(range(n_workers))
    # same semantics as map: the last worker to finish sees done=True
    assert sorted(done) == [False] * (n_workers - 1) + [True]


# ----------------------------------------------------------------
# key -> worker pinning and per-key serial order
# ----------------------------------------------------------------


def test_same_key_same_worker():
    n_workers = 4
    lock = threading.Lock()
    seen = {}

    def on_start(worker_info):
        return dict(worker_info=worker_info)

    def f(x, worker_info):
        with lock:
            seen.setdefault(x % 7, set()).add(worker_info.index)
        return x

    list(
        pl.thread.key_map(
            f,
            key=lambda x: x % 7,
            stage=range(200),
            workers=n_workers,
            on_start=on_start,
        )
    )

    # each key was handled by exactly one worker
    assert all(len(indexes) == 1 for indexes in seen.values())
    assert len(seen) == 7


def test_per_key_serial_order():
    n_workers = 4
    in_flight = set()
    lock = threading.Lock()
    order = []

    def f(x):
        k = x % 6
        with lock:
            assert k not in in_flight
            in_flight.add(k)
            order.append(x)
        time.sleep(0.001)
        with lock:
            in_flight.discard(k)
        return x

    data = list(range(120))
    result = list(
        pl.thread.key_map(f, key=lambda x: x % 6, stage=data, workers=n_workers)
    )

    assert sorted(result) == sorted(data)

    # elements of each key leave (and therefore enter) the handler in the
    # order in which they were received from upstream
    for k in range(6):
        assert [x for x in order if x % 6 == k] == [x for x in data if x % 6 == k]


def test_partitions_run_in_parallel():
    n_workers = 4
    active = 0
    max_active = 0
    lock = threading.Lock()

    def f(x):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return x

    list(
        pl.thread.key_map(
            f,
            key=lambda x: x % n_workers,
            stage=range(n_workers * 4),
            workers=n_workers,
        )
    )

    assert max_active == n_workers


# ----------------------------------------------------------------
# resource injection, worker_info, element_index
# ----------------------------------------------------------------


def test_on_start_on_done_per_worker():
    n_workers = 4
    started = []
    done = []
    lock = threading.Lock()

    def on_start(worker_info):
        with lock:
            started.append(worker_info.index)
        return dict(session=f"session-{worker_info.index}")

    def on_done(stage_status, session):
        with lock:
            done.append(session)
        assert stage_status.done in (True, False)

    def f(x, session):
        return (x, session)

    result = list(
        pl.thread.key_map(
            f,
            _key,
            range(50),
            workers=n_workers,
            on_start=on_start,
            on_done=on_done,
        )
    )

    assert sorted(started) == list(range(n_workers))
    assert sorted(done) == [f"session-{i}" for i in range(n_workers)]

    # the session injected by on_start is reused on every element of the
    # partition, i.e. state is shared by key within a worker
    for x, session in result:
        assert session == f"session-{hash(_key(x)) % n_workers}"


def test_worker_info_indexes():
    n_workers = 3

    def on_start(worker_info):
        return dict(worker_info=worker_info)

    indexes = set(
        pl.thread.key_map(
            lambda x, worker_info: worker_info.index,
            _key,
            range(60),
            workers=n_workers,
            on_start=on_start,
        )
    )

    assert indexes == set(range(n_workers))


def test_element_index_preserved():
    elements = list(
        pl.thread.to_iterable(
            pl.thread.key_map(lambda x: x * 2, _key, range(10), workers=3),
            return_index=True,
        )
    )

    # indexes stay attached to their elements even though partitions finish
    # in different orders
    by_index = {elem.index: elem.value for elem in elements}
    assert by_index == {(i,): i * 2 for i in range(10)}


# ----------------------------------------------------------------
# composition with native stages
# ----------------------------------------------------------------


@hp.given(nums=st.lists(st.integers()))
@hp.settings(max_examples=MAX_EXAMPLES)
def test_composes_with_native_map_and_filter(nums: tp.List[int]):
    stage = pl.thread.from_iterable(nums)
    stage = pl.thread.key_map(lambda x: x + 1, _key, stage, workers=3)
    stage = pl.thread.map(lambda x: x * 2, stage, workers=2)
    stage = pl.thread.filter(lambda x: x % 2 == 0, stage, workers=2)

    nums_pl = sorted(stage)
    nums_py = sorted(2 * (x + 1) for x in nums if (2 * (x + 1)) % 2 == 0)

    assert nums_pl == nums_py


def test_composes_with_each_sink():
    collected = []
    lock = threading.Lock()

    def sink(x):
        with lock:
            collected.append(x)

    pl.thread.each(
        sink,
        pl.thread.key_map(lambda x: x**2, _key, range(20), workers=3),
        workers=2,
        run=True,
    )

    assert sorted(collected) == sorted(x**2 for x in range(20))


# ----------------------------------------------------------------
# bounded buffering
# ----------------------------------------------------------------


def test_maxsize_bounds_buffering():
    n_workers = 3
    maxsize = 2
    in_stage = 0
    max_in_stage = 0
    lock = threading.Lock()
    processed = threading.Event()

    def f(x):
        nonlocal in_stage, max_in_stage
        with lock:
            in_stage += 1
            max_in_stage = max(max_in_stage, in_stage)
        time.sleep(0.005)
        with lock:
            in_stage -= 1
        if x == 0:
            processed.set()
        return x

    stage = pl.thread.key_map(
        f,
        key=lambda x: x % n_workers,
        stage=range(100),
        workers=n_workers,
        maxsize=maxsize,
    )
    stage = pl.thread.to_iterable(stage, maxsize=maxsize)

    iterator = iter(stage)
    first = next(iterator)
    processed.wait(2.0)
    time.sleep(0.05)

    # only one element per worker at a time (serial within a partition) and
    # the bounded queues cannot make the stage grow without bound
    assert max_in_stage <= n_workers

    # full pipeline still drains correctly (cross-partition order is unbound);
    # `first` was the single element already consumed above
    assert {first} | set(iterator) == set(range(100))


# ----------------------------------------------------------------
# failures and early consumer shutdown must not leave hung queues
# ----------------------------------------------------------------


class MyError(Exception):
    pass


def test_f_error_propagates():
    def f(x):
        if x == 7:
            raise MyError("boom")
        return x

    error = None
    try:
        list(pl.thread.key_map(f, _key, range(100), workers=4, maxsize=2))
    except MyError as e:
        error = e

    assert isinstance(error, MyError)


def test_key_error_propagates():
    def bad_key(x):
        if x == 5:
            raise MyError("bad key")
        return x

    error = None
    try:
        list(pl.thread.key_map(lambda x: x, bad_key, range(100), workers=4, maxsize=2))
    except MyError as e:
        error = e

    assert isinstance(error, MyError)


def test_unhashable_key_propagates():
    error = None
    try:
        list(
            pl.thread.key_map(
                lambda x: x,
                key=lambda x: [x],  # lists are unhashable
                stage=range(5),
                workers=2,
            )
        )
    except TypeError as e:
        error = e

    assert isinstance(error, TypeError)


def test_no_hung_threads_after_f_error():
    def f(x):
        time.sleep(0.002)
        if x == 3:
            raise MyError()
        return x

    before = threading.active_count()

    for _ in range(3):
        try:
            list(
                pl.thread.key_map(
                    f,
                    key=lambda x: x % 4,
                    stage=range(50),
                    workers=4,
                    maxsize=2,
                )
            )
        except MyError:
            pass

    time.sleep(0.3)

    assert threading.active_count() <= before + 1


def test_no_hung_threads_after_consumer_closes():
    before = threading.active_count()

    def slow(x):
        time.sleep(0.01)
        return x

    stage = pl.thread.key_map(
        slow,
        key=lambda x: x % 4,
        stage=range(1000),
        workers=4,
        maxsize=2,
    )

    for i, x in enumerate(stage):
        if i == 5:
            break

    time.sleep(0.3)

    # supervisor stopped every router/partition thread and all queues unblocked
    assert threading.active_count() <= before + 1


def test_rerunnable_after_failure():
    def f(x):
        if x == 2:
            raise MyError()
        return x

    for _ in range(2):
        try:
            list(pl.thread.key_map(f, _key, range(10), workers=3))
        except MyError:
            pass

    # a later run on a fresh stage works: context is recreated per build
    assert sorted(pl.thread.key_map(lambda x: x, _key, range(5), workers=3)) == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_timeout_restarts_partition_worker():
    def f(x):
        if x == 1:
            while True:
                time.sleep(0.02)
        return x

    result = list(
        pl.thread.key_map(
            f,
            key=lambda x: x % 3,
            stage=range(9),
            workers=3,
            timeout=0.3,
        )
    )

    # the stuck element is dropped when its worker is stopped and restarted
    assert sorted(result) == sorted(x for x in range(9) if x != 1)
