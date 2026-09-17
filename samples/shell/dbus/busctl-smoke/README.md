# sample: dbus/busctl-smoke

Smoke-test brain's D-Bus control surface with systemd's `busctl` alone - no
Python, no client library. `busctl_smoke.sh` starts `brain serve --dbus`
itself, then introspects the interface, reads properties, lists models,
fetches the manifests, and calls `Run` on the always-available `demo` model
(which returns its result as a file descriptor).

```bash
dbus-run-session -- bash samples/shell/dbus/busctl-smoke/busctl_smoke.sh target/debug/brain
```

## What it demonstrates

* The surface and its reply signatures are reachable with nothing but
  `busctl` - useful when jeepney (or Python at all) isn't available.
* Properties (`Version`, `ActiveJobs`, `Models`), `ListModels`, `Manifests`,
  an fd-returning `Run`, and `Cancel` on a bogus job id (`b false`).
* `busctl` is good for validating the surface itself; a real client (see
  `samples/python/dbus/brain-dbus/`) is needed to actually consume returned
  fds.

## What it needs

- `busctl` (systemd) and a private session bus - `dbus-run-session` wraps
  one so this needs no system D-Bus policy.
- A `brain` binary (`make build/debug`); pass its path as the one positional
  argument, default `target/debug/brain`.
- To validate against an already-running server instead of one this script
  starts and stops itself (what `tests/e2e/examples.bats` does, to avoid two
  processes racing for the same well-known bus name), set
  `BRAIN_DBUS_EXTERNAL=1`:

  ```bash
  BRAIN_DBUS_EXTERNAL=1 dbus-run-session -- bash samples/shell/dbus/busctl-smoke/busctl_smoke.sh
  ```
