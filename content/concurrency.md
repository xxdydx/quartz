
## process vs. thread

**Process:** An instance of a running program. It possesses an isolated virtual address space, file descriptors, and security context. Context switching between processes is expensive due to the need to flush the TLB (Translation Lookaside Buffer).

**Thread:** An independent unit of execution _within_ a process.
- **Shared State:** All threads in a process share the same Heap (dynamic memory), Global Variables, and Code Segment.
- **Private State:** Each thread has its own Stack (for local variables and function calls), Register Set (including Program Counter), and Thread Local Storage (TLS).
- **Implication:** Because threads share the same address space, they can communicate via shared memory extremely quickly, but this introduces the risk of data corruption.

