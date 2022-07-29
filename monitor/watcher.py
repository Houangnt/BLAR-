import time

import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from config import DIRECTORY_TO_WATCH, SERVICE_URL


class Watcher:
    def __init__(self, directory):
        self.directory = directory
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.directory, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except Exception as e:
            print(e)
            self.observer.stop()

        self.observer.join()


class Handler(FileSystemEventHandler):

    def on_any_event(self, event):
        if event.is_directory:
            return None
        elif event.event_type == 'closed':
            # print(event.src_path)
            send_request(event.src_path)


def send_request(file_path):
    s = requests.Session()
    resp = s.post(SERVICE_URL, json={'img_url': file_path})
    print("Status code: ", resp.status_code)
    print(resp.json())


if __name__ == '__main__':
    w = Watcher(DIRECTORY_TO_WATCH)
    w.run()
