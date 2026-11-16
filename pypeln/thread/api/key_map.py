import dataclasses
import threading
import time
import typing as tp
from dataclasses import dataclass, field

import stopit

from pypeln import utils as pypeln_utils
from pypeln.utils import A, B, T

from ..queue import IterableQueue, OutputQueues
from ..stage import Stage
from ..utils import Namespace
from ..worker import ApplyProcess, StageParams, Worker
from .to_stage import to_stage


class KeyFn(pypeln_utils.Protocol):
    def __call__(self, x: A) -> tp.Hashable:
        ...


class KeyedMapFn(pypeln_utils.Protocol):
    def __call__(self, x: A, **kwargs) -> B:
        ...


@dataclass
class KeyedContext:
    """Shared runtime state of a keyed stage, (re)created on every pipeline run."""

    input_queue: IterableQueue
    partition_queues: tp.List[IterableQueue]
    failed: threading.Event = field(default_factory=threading.Event)

    def fail(self):
        """Unblock every internal queue so no thread can be left waiting."""
        if self.failed.is_set():
            return

        self.failed.set()

        self.input_queue.kill()
        for queue in self.partition_queues:
            queue.kill()


@dataclass
class KeyedMap(ApplyProcess):
    f: KeyedMapFn
    context: tp.Optional[KeyedContext]

    def __call__(self, worker: Worker, **kwargs):
        for elem in worker.stage_params.input_queue:
            with worker.measure_task_time():
                self.apply(worker, elem, **kwargs)

        if self.context.failed.is_set():
            # the router or another partition already reported the exception.
            # Dont emit DONE: keeping the output queue open forces the consumer
            # to keep polling and actually observe the reported exception.
            raise pypeln_utils.StopThreadException

    def apply(self, worker: Worker, elem: tp.Any, **kwargs):
        if "element_index" in worker.f_args:
            kwargs["element_index"] = elem.index

        try:
            y = self.f(elem.value, **kwargs)
        except BaseException:
            # unblock the router, the other partitions and the upstream
            self.context.fail()
            raise

        worker.stage_params.output_queues.put(elem.set(y))


@dataclass
class Router:
    key: KeyFn
    n_partitions: int
    context: KeyedContext

    def __call__(self, worker: Worker, **kwargs):
        for elem in worker.stage_params.input_queue:
            if self.context.failed.is_set():
                # another thread already failed; stop draining the upstream
                return

            try:
                partition = hash(self.key(elem.value)) % self.n_partitions
            except BaseException:
                self.context.fail()
                raise

            self.context.partition_queues[partition].put(elem)

        # upstream exhausted: terminate every partition, even the empty ones
        for queue in self.context.partition_queues:
            queue.worker_done()


class RouterWorker(Worker):
    """
    The router owns the input queue and, unlike regular workers, never emits
    ``DONE`` itself: the partition workers are the sources of the output queues.
    """

    def __call__(self):
        try:
            self.process_fn(self)
        except pypeln_utils.StopThreadException:
            pass
        except BaseException as e:
            try:
                self.main_queue.raise_exception(e)
                time.sleep(0.001)
            except pypeln_utils.StopThreadException:
                pass
        finally:
            self.namespace.done = True

    def stop(self):
        if self.process is None:
            return

        self.namespace.task_start_time = None

        if not self.process.is_alive():
            return

        stopit.async_raise(
            self.process.ident,
            pypeln_utils.StopThreadException,
        )


@dataclass(eq=False)
class KeyedStage(Stage[T]):
    key: tp.Optional[KeyFn] = None

    def build(
        self,
        built: tp.Dict["Stage", OutputQueues],
        output_queue: IterableQueue,
        main_queue: IterableQueue,
    ) -> tp.Iterable[Worker]:
        if self in built:
            built[self].append(output_queue)
            return

        built[self] = OutputQueues([output_queue])

        input_queue = IterableQueue(
            maxsize=self.maxsize, total_sources=self.total_sources
        )

        # one bounded queue per partition keeps the whole stage bounded
        partition_queues = [
            IterableQueue(maxsize=self.maxsize, total_sources=1)
            for _ in range(self.workers)
        ]

        context = KeyedContext(
            input_queue=input_queue,
            partition_queues=partition_queues,
        )

        namespace = Namespace(active_workers=self.workers)

        for i, partition_queue in enumerate(partition_queues):
            stage_params = StageParams(
                input_queue=partition_queue,
                output_queues=built[self],
                namespace=namespace,
            )

            worker = Worker(
                process_fn=dataclasses.replace(self.process_fn, context=context),
                index=i,
                timeout=self.timeout,
                stage_params=stage_params,
                main_queue=main_queue,
                on_start=self.on_start,
                on_done=self.on_done,
                f_args=self.f_args,
            )

            yield worker

        router_stage_params = StageParams(
            input_queue=input_queue,
            output_queues=built[self],
            namespace=namespace,
        )

        router = RouterWorker(
            process_fn=Router(
                key=self.key,
                n_partitions=self.workers,
                context=context,
            ),
            index=-1,
            timeout=0,
            stage_params=router_stage_params,
            namespace=Namespace(done=False, task_start_time=None),
            main_queue=main_queue,
            on_start=None,
            on_done=None,
            f_args=[],
        )

        yield router

        for dependency in self.dependencies:
            yield from dependency.build(built, input_queue, main_queue)


@tp.overload
def key_map(
    f: KeyedMapFn,
    key: KeyFn,
    stage: tp.Union[Stage[A], tp.Iterable[A]],
    workers: int = 1,
    maxsize: int = 0,
    timeout: float = 0,
    on_start: tp.Callable = None,
    on_done: tp.Callable = None,
) -> Stage[B]:
    ...


@tp.overload
def key_map(
    f: KeyedMapFn,
    key: KeyFn,
    workers: int = 1,
    maxsize: int = 0,
    timeout: float = 0,
    on_start: tp.Callable = None,
    on_done: tp.Callable = None,
) -> pypeln_utils.Partial[Stage[B]]:
    ...


def key_map(
    f: KeyedMapFn,
    key: KeyFn,
    stage: tp.Union[
        Stage[A], tp.Iterable[A], pypeln_utils.Undefined
    ] = pypeln_utils.UNDEFINED,
    workers: int = 1,
    maxsize: int = 0,
    timeout: float = 0,
    on_start: tp.Callable = None,
    on_done: tp.Callable = None,
) -> tp.Union[Stage[B], pypeln_utils.Partial[Stage[B]]]:
    """
    Creates a stage that maps a function `f` over the data while pinning every
    element with the same key to the same worker for the whole run.

    ```python
    import pypeln as pl

    def key(x):
        return x["id"]

    def clean(record, session):
        return session.clean(record)

    stage = pl.thread.key_map(
        clean, key, records, workers=4, on_start=open_session
    )
    ```

    Guarantees:

    * All elements sharing a key are processed by the same worker, so the
      resources created by `on_start` and any worker local state are reused.
    * Elements of the same partition are processed serially in the order in
      which the router receives them from upstream. Different partitions keep
      running in parallel.
    * The stage is bounded by `maxsize`; a `maxsize` of `0` (default) is
      unbounded, exactly like `map`.
    * On a normal end of input every upstream element and every partition,
      including the empty ones, is drained and closed before the stage ends.
    * If `key` or `f` raise, or the consumer closes the pipeline early, all the
      internal queues are unblocked so no thread is left hanging.

    Arguments:
        f: A function with the signature `f(x) -> y`. `f` can accept special
            additional arguments by name as described in [Advanced Usage](https://cgarciae.github.io/pypeln/advanced/#dependency-injection).
        key: A function with the signature `key(x) -> hashable` used to select
            the worker that processes `x`. The same key always maps to the same
            worker during a run.
        stage: A Stage or Iterable.
        workers: The number of workers (partitions) the stage should contain.
        maxsize: The maximum number of objects the stage can hold
            simultaneously, if set to `0` (default) then the stage can grow
            unbounded.
        timeout: Seconds before stoping a worker if its current task is not yet
            completed. Defaults to `0` which means its unbounded.
        on_start: A function with signature `on_start(worker_info?) -> kwargs?`,
            where `kwargs` can be a `dict` of keyword arguments that can be
            consumed by `f` and `on_done`. `on_start` can accept additional
            arguments by name as described in [Advanced Usage](https://cgarciae.github.io/pypeln/advanced/#dependency-injection).
        on_done: A function with signature `on_done(stage_status?)`. This
            function is executed once per worker when the worker finishes.
            `on_done` can accept additional arguments by name as described in
            [Advanced Usage](https://cgarciae.github.io/pypeln/advanced/#dependency-injection).

    Returns:
        Returns a `Stage` if the `stage` parameter is given, else it returns a
        `Partial`.
    """

    if isinstance(stage, pypeln_utils.Undefined):
        return pypeln_utils.Partial(
            lambda stage: key_map(
                f,
                key,
                stage=stage,
                workers=workers,
                maxsize=maxsize,
                timeout=timeout,
                on_start=on_start,
                on_done=on_done,
            )
        )

    stage = to_stage(stage, maxsize=maxsize)

    return KeyedStage(
        process_fn=KeyedMap(f=f, context=None),
        key=key,
        workers=workers,
        maxsize=maxsize,
        timeout=timeout,
        total_sources=stage.workers,
        dependencies=[stage],
        on_start=on_start,
        on_done=on_done,
        f_args=pypeln_utils.function_args(f),
    )
