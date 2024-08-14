import sys
import inspect
import asyncio
import traceback
import multiprocessing

import _thread
from threading import Timer
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from loguru import logger
from neurons.utils.log import configure_logging
from neurons.validator.config.lists import get_warninglist


# Background Loop
class BackgroundTimer(Timer):
    def __str__(self) -> str:
        return self.function.__name__

    def run(self):
        configure_logging()
        self.function(*self.args, **self.kwargs)
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


class MultiprocessBackgroundTimer(multiprocessing.Process):
    def __str__(self) -> str:
        return self.function.__name__

    def __init__(self, interval, function, args=None, kwargs=None, timeout=300):
        super().__init__()
        self.interval = interval
        self.function = function
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}
        self.finished = multiprocessing.Event()
        self.timeout = timeout

    def run_with_timeout(self, func, *args, **kwargs):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=self.timeout)
            except TimeoutError:
                logger.error(
                    f"[thread] {self.function.__name__} timed out"
                    + f" after {self.timeout} seconds"
                )
                # Optionally, you might want to kill the thread here
                # This is a bit tricky and might not always work as expected
                return None

    async def run_async_with_timeout(self, func, *args, **kwargs):
        try:
            return await asyncio.wait_for(
                func(*args, **kwargs), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.error(
                f"[thread] {self.function.__name__} timed out"
                + f" after {self.timeout} seconds"
            )
            return None

    def run(self):
        configure_logging()

        logger.info(f"[thread] {self.function.__name__} started")

        while not self.finished.is_set():
            try:
                if inspect.iscoroutinefunction(self.function):
                    asyncio.run(
                        self.run_async_with_timeout(
                            self.function, *self.args, **self.kwargs
                        )
                    )
                else:
                    self.run_with_timeout(
                        self.function, *self.args, **self.kwargs
                    )
                self.finished.wait(self.interval)
            except Exception:
                logger.error(traceback.format_exc())

    def cancel(self):
        logger.info(f"[thread] cancel {self.function.__name__}")
        self.finished.set()


def get_coldkey_for_hotkey(self, hotkey):
    """
    Look up the coldkey of the caller.
    """
    if hotkey in self.metagraph.hotkeys:
        index = self.metagraph.hotkeys.index(hotkey)
        return self.metagraph.coldkeys[index]
    return None


background_steps: int = 0


def background_loop(self, is_validator: bool):
    """
    Handles terminating the miner after deregistration and
    updating the blacklist and whitelist.
    """
    from neurons.constants import IS_CI_ENV

    if IS_CI_ENV:
        return

    global background_steps
    background_steps += 1

    # Terminate the miner / validator after deregistration
    if background_steps % 5 != 0:
        return

    neuron_type = "Validator" if is_validator else "Miner"

    my_hotkey: str = self.wallet.hotkey.ss58_address
    hotkeys, _coldkeys = asyncio.run(get_warninglist())

    try:
        if my_hotkey in hotkeys.keys():
            hotkey_warning: str = hotkeys[my_hotkey][1]

            logger.info(
                f"This hotkey is on the warning list: {my_hotkey}"
                + f" | Date for rectification: {hotkey_warning}",
            )

        self.metagraph.sync(subtensor=self.subtensor)
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            logger.info(f">>> {neuron_type} has deregistered... terminating.")
            try:
                _thread.interrupt_main()
            except Exception as e:
                logger.info(
                    f"An error occurred trying to terminate the main thread: {e}."
                )

            sys.exit(0)

    except Exception as e:
        logger.error(
            f">>> An unexpected error occurred syncing the metagraph: {e}"
        )


def normalize_weights(weights):
    sum_weights = float(sum(weights))
    normalizer = 1 / sum_weights
    weights = [weight * normalizer for weight in weights]
    if sum(weights) < 1:
        diff = 1 - sum(weights)
        weights[0] += diff

    return weights
