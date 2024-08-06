import os
import sys
import inspect
import asyncio
import traceback
import multiprocessing
from threading import Timer
from loguru import logger

from neurons.common_schema import NeuronAttributes
from neurons.utils.log import configure_logging


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

    def __init__(self, interval, function, args=None, kwargs=None):
        super().__init__()
        self.interval = interval
        self.function = function
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}
        self.finished = multiprocessing.Event()

    def run(self):
        configure_logging()

        logger.info(f"{self.function.__name__} started")

        while not self.finished.is_set():
            try:
                if inspect.iscoroutinefunction(self.function):
                    asyncio.run(self.function(*self.args, **self.kwargs))
                else:
                    self.function(*self.args, **self.kwargs)

                self.finished.wait(self.interval)

            except Exception:
                logger.error(traceback.format_exc())

    def cancel(self):
        self.finished.set()


def send_run_command(command_queue, command, data):
    """
    Send a command to the main process with the associated data.
    """
    command_queue.put((command, data))
    logger.info(f"Sent command: {command} with data: {data}")


def kill_main_process_if_deregistered(
    command_queue,
    neuron_attributes: NeuronAttributes,
):
    # Terminate the miner / validator after deregistration
    if (
        neuron_attributes.background_steps % 5 == 0
        and neuron_attributes.background_steps > 1
    ):
        try:
            if (
                neuron_attributes.wallet_hotkey_ss58_address
                not in neuron_attributes.hotkeys
            ):
                logger.info(">>> Neuron has deregistered... terminating.")
                try:
                    send_run_command(command_queue, "die", None)
                except Exception as e:
                    logger.info(
                        f"An error occurred trying to terminate the main thread: {e}."
                    )
                try:
                    os.exit(0)
                except Exception as e:
                    logger.error(
                        f"An error occurred trying to use os._exit(): {e}."
                    )
                sys.exit(0)
        except Exception as e:
            logger.error(
                f">>> An unexpected error occurred syncing the metagraph: {e}"
            )
