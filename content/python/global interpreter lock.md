
## how python threads work
- Python uses native OS threads (POSIX/pthreads on UNIX, windows threads). the interpreter is fully threaded.
- The GIL exists to simplify the C implementation of the Python interpreter, protecting memory management and C extensions from race conditions.
- However, the GIL prevents true parallel computing. Threads must take turns as a result. Normally, a thread yields the GIL voluntarily when it does I/O (like reading a file or waiting on a network socket).
- **The 100-Tick Check:** If a thread is purely CPU-bound, Python forces it to periodically check in. By default, this happens every 100 "ticks" (roughly 100 bytecode instructions).
- **The Context Switch:** Every 100 ticks, the running thread runs a simple C function that drops the GIL and immediately tries to acquire it back. 

