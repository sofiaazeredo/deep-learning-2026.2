"""Relatório PASS/WARN/FAIL compartilhado pelos checks."""
import sys


class Report:
    def __init__(self, title):
        self.title, self.rows = title, []
        print(f"\n===== {title} =====")

    def _add(self, st, name, detail):
        self.rows.append((st, name, detail))
        print(f"[{st}] {name}" + (f" — {detail}" if detail != "" else ""), flush=True)

    def ok(self, name, detail=""):
        self._add("PASS", name, detail)

    def warn(self, name, detail=""):
        self._add("WARN", name, detail)

    def fail(self, name, detail=""):
        self._add("FAIL", name, detail)

    def check(self, cond, name, detail="", warn_only=False):
        (self.ok if cond else (self.warn if warn_only else self.fail))(name, detail)
        return bool(cond)

    def info(self, text):
        print(f"       {text}", flush=True)

    def finish(self):
        n = {k: sum(r[0] == k for r in self.rows) for k in ("PASS", "WARN", "FAIL")}
        print(f"\n{self.title}: {n['PASS']} PASS, {n['WARN']} WARN, {n['FAIL']} FAIL")
        if n["FAIL"]:
            sys.exit(1)
