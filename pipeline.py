import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import cv2
from app.controller.BLPR import get_algorithm
import queue
from app.extensions import config, logger
import shutil
import os
from datetime import datetime

class Watcher:
    def __init__(self, directory, queue):
        self.observer = Observer()
        self.directory = directory
        self.queue = queue

    def run(self):
        event_handler = FileSystemEventHandler()
        event_handler.on_any_event = self.on_event
        self.observer.schedule(event_handler, self.directory, recursive=True)
        self.observer.start()
        try:
            logger.info(f"Watcher starts watching at {self.directory}")
            while True:
                time.sleep(2)
        except Exception as e:
            self.observer.stop()
            logger.error(e)
            logger.error("Observer stopped")
        self.observer.join()

    def on_event(self, event):
        if event.is_directory:
            return None
        if event.event_type == 'closed':
            self.queue.put(event.src_path)


def start_watcher(watcher):
    watcher.run()


def worker(queue, algorithm):
    logger.info(f"Worker started!")
    while True:
        new_file = queue.get()
        if new_file:
            try:
                timestamp = datetime.today().strftime('%y%m%d')
                if not os.path.isdir(f"{config.DUMP_DIR}/{timestamp}"):
                    os.mkdir(f"{config.DUMP_DIR}/{timestamp}")
                logger.info(f"[START] processing {new_file}")
                start_time = time.time()
                img = cv2.imread(new_file)
                results = algorithm.process(img)
                if results.barcodes:
                    dest_filename = f"{os.path.splitext(os.path.basename(new_file))[0]}_{'_'.join(results.barcodes)}.jpg"
                else:
                    dest_filename = f"{os.path.splitext(os.path.basename(new_file))[0]}_noread.jpg"
                shutil.copy(new_file, f"{config.DUMP_DIR}/{timestamp}/{dest_filename}")
                os.remove(new_file)
                logger.info(f"[FINISH] processing {new_file}")
                logger.info(f"[PROCESSING TIME] of {new_file} is {time.time() - start_time} seconds")
            except Exception as e:
                logger.error(e)
                logger.error(f"Error processing {new_file}")


if __name__ == "__main__":
    q = queue.Queue()

    workers = []
    algorithm = get_algorithm()
    logger.info('Algorithm instance created!')

    for i in range(config.NUM_OF_WORKERS):
        t = threading.Thread(target=worker, args=(q, algorithm))
        workers.append(t)
        t.start()

    watcher = Watcher(config.INPUT_DIR, q)
    producer = threading.Thread(target=start_watcher(watcher), daemon=True)
    producer.start()

    for t in workers:
        t.join()
    producer.join()
    q.join()
