The **OSI model** breaks the complexity of networking into **7 layers**, each with a specific job. This allows for abstraction and separation of concerns.

| Layer | Name         | Job                             | Example              |
| ----- | ------------ | ------------------------------- | -------------------- |
| 7     | Application  | What the user/app sees          | HTTP, DNS, SMTP      |
| 6     | Presentation | Formatting, encryption          | TLS, JPEG            |
| 5     | Session      | Managing connections            | Login sessions       |
| 4     | Transport    | Reliable delivery, ports        | TCP, UDP             |
| 3     | Network      | Routing across networks         | IP addresses         |
| 2     | Data Link    | Delivery on one network segment | Ethernet, MAC        |
| 1     | Physical     | Raw bits on a wire              | Cables, WiFi signals |

Most network engineers only care about layers 2 to 4.
#### How data flows:
When you send data, it travels **down** the layers on your machine (each layer wraps the data with its own header — this is called **encapsulation**), gets transmitted physically, then travels **up** the layers on the receiver's machine (each layer unwraps its header — **decapsulation**).

```bash
Sender:                          Receiver:
[App data]                       [App data]
[L4 header | App data]    →      [L4 header | App data]
[L3 header | L4 | App]           [L3 header | L4 | App]
[L2 header | L3 | L4 | App]      [L2 header | L3 | L4 | App]
[bits on wire]          →→→→→→→  [bits on wire]
```

Each layer only reads its own header and passes the rest up. Layer 3 doesn't know what's inside the TCP packet. TCP doesn't know what's inside the HTTP request.


## Layer 1 — The Physical Layer 

This is the layer that actually moves **bits** — 1s and 0s — from one place to another as electrical signals, light pulses, or radio waves.

**Key concepts**:

**Bandwidth** — how many bits per second (bps) can travel through a medium. Modern networks talk in Gbps (gigabits per second). Data center links are often 10, 25, 100, or 400 Gbps.

**Latency** — delay. Even at the speed of light, distance adds up. A signal from Singapore to New York takes ~170ms just from physics. You can't engineer your way around the speed of light.


**Transmission media:**

- **Copper (Ethernet cable)** — cheap, short distances (up to ~100m). The RJ45 cable you plug into your laptop.
- **Fiber optic** — light through glass. Very fast, very long distances, used in data centers and undersea cables connecting continents.
- **WiFi (wireless)** — radio waves. Convenient but shared medium, more interference, higher latency than wired.

**Half-duplex vs Full-duplex:**

- **Half-duplex** — can send OR receive at a time, not both (like a walkie-talkie)
- **Full-duplex** — can send AND receive simultaneously (like a phone call). Modern ethernet is full-duplex.

## Layer 2 — Data Link Layer

Layer 2 is responsible for getting data from **one device to the next device** on the **same network segment**. Not across the internet — just the next hop.

The unit of data at Layer 2 is called a **frame**.

#### MAC Addresses
Every network interface (your WiFi card, ethernet port) has a **MAC address** — a unique 48-bit hardware identifier, usually written like `00:1A:2B:3C:4D:5E`.

- **IP addresses** (Layer 3) identify _where_ you are logically on the internet — they can change
- **MAC addresses** (Layer 2) identify the _physical device_ — they're burned into hardware

Analogy: IP address is like a mailing address (can change if you move), but a MAC address is like an NRIC number.

When your laptop sends data to your router, it wraps the packet in a frame addressed to the router's MAC address. The router strips the frame, reads the IP packet inside, and handles it at Layer 3.

#### Ethernet
**Ethernet** is the dominant Layer 2 protocol. It defines how frames are structured and how devices share a cable/switch.

A basic Ethernet frame looks like:
```
| Destination MAC | Source MAC | Type | Payload (IP packet) | FCS |
```

- **Destination MAC** — who this frame is for
- **Source MAC** — who sent it
- **Type** — what's inside (usually IPv4 or IPv6)
- **FCS** — a checksum to detect corruption


#### Switches

A **switch** is a Layer 2 device that connects multiple devices on the same network and forwards frames between them intelligently.

When a frame arrives, the switch reads the **destination MAC address** and forwards it only to the correct port — not to everyone. This is smarter than an older device called a **hub**, which just blasted every frame to every port (noisy and inefficient).

**How does a switch know which port to send to?** It builds a **MAC address table** by learning:

```
"Port 3 sent me a frame with source MAC AA:BB:CC... 
so AA:BB:CC must be reachable via Port 3"
```

Over time it builds a full map. If it doesn't know a destination MAC yet, it **floods** the frame to all ports and learns from the response. This is called **unknown unicast flooding**.

**Broadcast Domains**

Some frames are sent to **everyone** — these are **broadcast** frames (destination MAC = `FF:FF:FF:FF:FF:FF`). All devices that receive a broadcast are in the same **broadcast domain**.

Problem: if you have 1000 devices on the same network, every broadcast hits all 1000. This gets noisy and wastes bandwidth. This is why you split networks up.


#### VLANs (Virtual Local Area Networks)

A **VLAN** lets you logically split one physical switch into multiple isolated networks. Devices on VLAN 10 can't talk directly to devices on VLAN 20, even if they're plugged into the same switch.

```
Physical switch:
├── Port 1, 2, 3  → VLAN 10 (e.g. Engineering)
├── Port 4, 5, 6  → VLAN 20 (e.g. Finance)
└── Port 7        → Trunk (carries all VLANs to router)
```

**Why VLANs matter:**
- **Security** — isolate sensitive systems
- **Performance** — smaller broadcast domains
- **Flexibility** — reorganize networks without rewiring

A **trunk port** carries traffic from multiple VLANs between switches, tagging each frame with a VLAN ID (802.1Q standard) so the receiving switch knows which VLAN it belongs to.

## Layer 3 — Network Layer

Layer 3 is responsible for getting data **across multiple networks** — from a laptop all the way to a server on the other side of the world. The unit of data here is a **packet**.

#### IP Addresses

An **IP address** (IPv4) is a 32-bit number, written as four octets: `192.168.1.10`

Every device on a network has one. Unlike MAC addresses, IP addresses are **logical** — they're assigned, not burned in. They encode **location** in the network hierarchy.

**IPv4 vs IPv6:**

- IPv4: 32-bit, ~4 billion addresses — we've run out
- IPv6: 128-bit, effectively unlimited — slowly replacing IPv4

#### Packets

A Layer 3 packet contains:

```
| Source IP | Destination IP | TTL | Protocol | Payload (TCP/UDP data) |
```

- **TTL (Time to Live)** — decremented at each router. Hits 0? Packet is dropped. Prevents packets looping forever.
- **Protocol** — tells the receiver what's inside (TCP = 6, UDP = 17)

#### Routers

A **router** connects different networks and forwards packets between them based on destination IP. Unlike a switch (which uses MACs and operates within one network), a router operates across networks.

When a packet arrives, the router:

1. Reads the destination IP
2. Looks it up in its **routing table**
3. Forwards the packet out the correct interface toward the destination

```
Routing table example:
Destination          Next Hop        Interface
10.0.0.0/8          10.1.1.1        eth0
192.168.1.0/24      directly conn.  eth1
0.0.0.0/0           203.1.1.1       eth2   ← default route
```

The **default route** (`0.0.0.0/0`) means: "if nothing else matches, send it here." It's how your home router sends everything to your ISP.


Here's a full overview of how a packet travels across the internet.

```
Your laptop
  → wraps data in TCP segment (L4)
  → wraps in IP packet with destination IP (L3)
  → wraps in Ethernet frame to router's MAC (L2)
  → sends bits on wire (L1)

Router 1: strips L2, reads L3 destination IP, re-wraps in new frame, forwards
Router 2: same
Router N: same
...
Destination server: unwraps all layers, reads data
```

