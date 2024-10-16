import concurrent.futures
import random

def worker(number):
    print("Worker %s: ", name)

pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

a = [random.randint(1, 100) for _ in range(100)]
file_queue = queue.Queue()

for x in a:
    file_queue.put(x)

pool.submit(worker)
pool.submit(worker)

pool.shutdown(wait=True)

print("Main thread continuing to run")