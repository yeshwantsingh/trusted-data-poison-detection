"""Delete detect.json files produced by an older detect.py (no alpha_sweep) and
mark their queue locks FAILED so the worker re-runs them."""
import glob, hashlib, json, os, time
n = 0
for line in open("queue/detect.txt"):
    done, dep, cmd = line.rstrip("\n").split("\t")
    if os.path.exists(done):
        d = json.load(open(done))
        if not any("alpha_sweep" in r for r in d["results"]):
            os.remove(done)
            lk = os.path.join("queue/locks", hashlib.md5(cmd.encode()).hexdigest())
            open(lk, "w").write(f"FAILED 1 {time.ctime()}\n{cmd}\n")
            n += 1
print("invalidated", n)
