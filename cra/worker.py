"""File-based task queue. Each line of a task file is `<done_marker>\t<command>`.
A task is skipped if its done marker exists, and claimed atomically via a lock
file, so several PBS jobs can drain the same queue concurrently.

python -m cra.worker queue/tasks_main.txt [more files...] --slots-per-gpu 2
"""
import argparse, hashlib, os, subprocess, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCKS = os.path.join(ROOT, "queue", "locks")


def tasks(files):
    for f in files:
        for line in open(f):
            line = line.rstrip("\n")
            if line and not line.startswith("#"):
                parts = line.split("\t")
                done, cmd = parts[0], parts[-1]
                dep = parts[1] if len(parts) == 3 else None
                yield os.path.join(ROOT, done), (os.path.join(ROOT, dep) if dep else None), cmd


def claim(done, cmd):
    if os.path.exists(done):
        return False
    lock = os.path.join(LOCKS, hashlib.md5(cmd.encode()).hexdigest())
    me = os.environ.get("PBS_JOBID", "local")
    if os.path.exists(lock):
        first = open(lock).readline().split()
        owner = first[0] if first else ""
        tries = int(first[1]) if len(first) > 1 and first[1].isdigit() else 1
        # a lock from an earlier (finished) job, or released after a failure, is reclaimable
        if (owner != me or owner == "FAILED") and tries < 3:
            open(lock, "w").write(f"{me} {tries + 1} {time.ctime()}\n{cmd}\n")
            return True
        return False
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{me} 1 {time.ctime()}\n{cmd}\n".encode()); os.close(fd)
        return True
    except FileExistsError:
        return False


RUNNING = [0]


def slot(gpu, files, lock):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    while True:
        with lock:
            waiting = False
            job = None
            for d, dep, c in tasks(files):
                if dep and not os.path.exists(dep):
                    if not os.path.exists(d):
                        waiting = True
                    continue
                if claim(d, c):
                    job = (d, c); RUNNING[0] += 1; break
            idle = RUNNING[0] == 0
        if job is None:
            if waiting and not idle:
                time.sleep(30); continue
            return
        done, cmd = job
        t0 = time.time()
        r = subprocess.run(cmd, shell=True, cwd=ROOT, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        status = "OK" if r.returncode == 0 else f"FAIL({r.returncode})"
        print(f"[gpu{gpu}] {status} {time.time()-t0:.0f}s {cmd}", flush=True)
        if r.returncode != 0:
            print(r.stdout[-3000:], flush=True)
            lk = os.path.join(LOCKS, hashlib.md5(cmd.encode()).hexdigest())
            first = open(lk).readline().split()
            open(lk, "w").write(f"FAILED {first[1] if len(first) > 1 else 1} {time.ctime()}\n{cmd}\n")
        with lock:
            RUNNING[0] -= 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--slots-per-gpu", type=int, default=2)
    a = p.parse_args()
    os.makedirs(LOCKS, exist_ok=True)
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    lock = threading.Lock()
    th = [threading.Thread(target=slot, args=(g, a.files, lock)) for g in gpus for _ in range(a.slots_per_gpu)]
    for t in th: t.start()
    for t in th: t.join()
    print("queue drained", flush=True)


if __name__ == "__main__":
    main()
