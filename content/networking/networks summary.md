trying to learn the basics for my internship :)


## OSI model

the **OSI model** breaks the complexity of networking into **7 layers**, each with a specific job.

| Layer | Name         | Job                             | Example              |
| ----- | ------------ | ------------------------------- | -------------------- |
| 7     | Application  | What the user/app sees          | HTTP, DNS, SMTP      |
| 6     | Presentation | Formatting, encryption          | TLS, JPEG            |
| 5     | Session      | Managing connections            | Login sessions       |
| 4     | Transport    | Reliable delivery, ports        | TCP, UDP             |
| 3     | Network      | Routing across networks         | IP addresses         |
| 2     | Data Link    | Delivery on one network segment | Ethernet, MAC        |
| 1     | Physical     | Raw bits on a wire              | Cables, WiFi signals |
in practice, 
"layer 3" = ip routing,
"layer 4" = tcp/udp,
"layer 7" = application (http)

## TCP/IP stack

TCP/IP is what the internet actually uses, 4 layers not 7. it basically collapses OSI's topmost 3 layers into 1 application layer.

- Application — HTTP, DNS, SMTP, SSH (OSI layers 5–7)
- Transport — TCP and UDP (OSI layer 4)
- Internet — IP, ICMP, routing (OSI layer 3)
- Network Access — Ethernet, Wi-Fi, physical (OSI layers 1–2)

as data moves down the stack, each layer wraps it in a header.

Application data → TCP segment → IP packet → Ethernet frame. 

at the destination, each layer strips its header and passes the payload up.


<u>example</u>

scenario: you sending a "Hello!" message to a friend via a web app (HTTP).
final data on the wire as bits after passing through the different layers in TCP/IP

```
[ Ethernet header: MAC_you → MAC_router | IP packet | TCP segment | "Hello!" ]
```



## IP Addressing & Subnets

IPv4 address is 32 bits, written as 4 octets. Two parts, network portion (identifying the subnet) and host portion (identifies the device). The subnet mask determines which bits belong to which part.

**Subnet masks & CIDR:** A mask of 255.255.255.0 = the first 24 bits are the network portion, written as /24. So 192.168.1.0/24 means up to 254 hosts (.1 to .254). A /25 splits that in half.

Common CIDR sizes:
- /24 → 256 addresses (254 usable) — single office/VLAN
- /22 → 1,024 addresses
- /16 → 65,536 addresses
- /30 → 4 addresses (2 usable) — point-to-point links
- /32 → single host

In any subnet, 2 addresses are always reserved and can't be assigned to devices.
- **Network address** — all host bits set to 0. Identifies the subnet itself. E.g. 192.168.1.**0**
- **Broadcast address** — all host bits set to 1. Sends to every device on the subnet. E.g. 192.168.1.**255**

**public vs private addresses**

private addresses are reserved for use inside networks. routers on the internet will never forward them:

|Range|CIDR|Typical use|
|---|---|---|
|10.0.0.0 – 10.255.255.255|/8|large enterprise, cloud|
|172.16.0.0 – 172.31.255.255|/12|medium networks|
|192.168.0.0 – 192.168.255.255|/16|home/small office|

public addresses are globally unique and what the internet actually routes. your ISP assigns you one.

other special ranges:

- **127.0.0.1** — loopback / localhost. always refers to your own machine, never leaves your device
- **169.254.x.x** — auto-assigned when DHCP fails. seeing this usually means something's wrong

**how addresses get assigned**

- DHCP (dynamic) — router hands out IPs automatically from a pool when devices connect
- static — manually configured. used for servers and network gear that need a permanent address

### Default Gateway

the gateway is the device that connects your local network to another network (usually the internet). it's the exit point for any traffic not destined for your local network.

```
Laptop  ─┐
Phone   ─┤──→  Router/Gateway  ──→  ISP  ──→  Internet
Tablet  ─┘     192.168.1.1
```

your router is your default gateway, typically at 192.168.1.1. every device is configured to say: "if i don't know where to send this packet, send it to 192.168.1.1 and let it figure it out."


to decide whether local or gateway, your device checks whether the destination is on the same subnet or not (using subnet mask):

```
Your IP:   192.168.1.5  /24  →  local range is 192.168.1.0 – 192.168.1.255

Destination 192.168.1.20?  → local, deliver directly
Destination 8.8.8.8?       → not local, send to gateway
```

**every device needs 3 things configured:**

|Setting|Example|Purpose|
|---|---|---|
|IP address|192.168.1.5|identifies your device|
|Subnet mask|255.255.255.0|defines what counts as "local"|
|Default gateway|192.168.1.1|where to send everything else|

without the gateway configured, a device can talk locally but has no way to reach the internet.

## Network Address Translation (NAT)

IPv4 only has ~4b addresses, NAT allows many devices to share one public IP. home router has 1 public IP address serving dozens of private devices, having a private IP of 192.168.x.x.

when a device sends a packet, the router rewrites the source IP to its public IP and records the mapping (including port) in a NAT table. for the reply, it reverses the translation and forwards to the original device.

for example, 

1. my laptop requests google.com. my laptop sends a packet as follows:
```
From: 192.168.1.2 : port 54321
To:   142.250.80.46 : port 443
```


2. packet hits the router, router knows the source is a private IP so it will rewrite the source in the packet.
```
From: 203.0.113.10 : port 54321   ← swapped to public IP
To:   142.250.80.46 : port 443
```

3. router records the mapping in its NAT table
```
203.0.113.10:54321 ↔ 192.168.1.2:54321
```

when reply from google.com arrives aty the router, router will check the NAT table and rewrites the desitnation back to the private IP.

if two devices make requests to the same source at the same time, the router uses the port number to tell them apart. each new connection the device makes gets a fresh random port number.

## Routing & Forwarding

```
Your laptop                                          Web server
192.168.1.5  →  Router A  →  Router B  →  Router C  →  8.8.8.8
```

let's say a packet starts at your laptop. it doesn't know how to get to the destination web server. it just sends to router A (your gateway). now router A has to decide, which direction shd i send this towards my destination? that decision is made using a routing table.


**the routing table**

every router maintains a routing table — essentially a list of rules:

```
Destination          Send to (next hop)
─────────────────────────────────────────
192.168.1.0/24       deliver locally
10.0.0.0/8           → Router B
0.0.0.0/0            → Router C     (default — "everything else")
```

when a packet arrives, the router looks up the destination IP in this table and sends it to the next hop. the next router does the same thing, and so on, until the packet reaches its destination.

**longest prefix match** just means: if multiple rules match, use the most specific one. A packet to 10.0.1.5 matches both 10.0.0.0/8 and 10.0.1.0/24 — the router picks /24 because it's more specific (narrower range = more precise instruction).

**routing vs forwarding**

routing — control plane: building the routing table, deciding which paths exist.
forwarding — data plane: for this packet, look up the destination IP and send out the right port/interface.

**how the routing table gets built**
- static routing — human types in every route.
- dynamic routing — routers talk to each other automatically, sharing information about what they can reach. if a link goes down, they detect it and reroute automatically. two main protocols:
	- OSPF — used inside a single organisation's network
	- BGP — used between organisations on the internet.


## Domain Name System (DNS)

it helps translate human-readable names (e.g. "google.com") into IP addresses. 

**resolution process**:
1. OS checks local cache
2. Asks your recursive resolver (ISP or 8.8.8.8)
3. Resolver asks a root nameserver → returns TLD server for .com
4. Asks the .com TLD nameserver → returns Google's authoritative nameserver
5. Asks Google's nameserver → returns the IP
6. Resolver caches the result (per TTL) and returns it to you

**key record types:**
- A — hostname → IPv4
- AAAA — hostname → IPv6
- CNAME — alias from one name to another
- MX — mail server for a domain
- NS — authoritative nameservers for a domain
- PTR — reverse lookup (IP → hostname)
- TTL — how long a record is cached (seconds)


### Packets, Frames & Segments

each layer has its own term for its unit of data:

- **segment** — transport layer (TCP/UDP)
- **packet** — network layer (IP)
- **frame** — data link layer (Ethernet)

key header fields to know (these are the headers found in flow records):

| Header         | Key fields                                                  |
| -------------- | ----------------------------------------------------------- |
| TCP segment    | source port, dest port, sequence number, flags, window size |
| IP packet      | source IP, dest IP, TTL, protocol (6=TCP, 17=UDP, 1=ICMP)   |
| Ethernet frame | source MAC, dest MAC                                        |

## Ports

**well-known ports (0–1023)** — fixed services, always the destination port when connecting to a server:

```
80    → HTTP
443   → HTTPS
53    → DNS
22    → SSH
25    → SMTP
```

**ephemeral ports (1024–65535)** — randomly assigned by OS for your side of a connection. thrown away when connection closes.

## Protocols


### TCP
- connection-oriented, before any data is sent, two devices must first establish a connection via a 3-way handshake.
```
Client  →  SYN        →  Server   "I want to connect"
Client  ←  SYN-ACK    ←  Server   "OK, I acknowledge"
Client  →  ACK        →  Server   "Great, connected"
```
- only after this is complete does data start flowing. when done, a 4-way teardown closes the connection (FIN, FIN-ACK, FIN, FIN-ACK).

TCP guarantees delivery by:
- giving every byte a **sequence number**
- receiver sends back **ACKs** (acknowledgements) confirming what it received
- if no ACK arrives in time, sender **retransmits** the packet
- packets that arrive out of order get **reordered** before passing to the application

TCP also does **congestion control** — it monitors the network and slows down if it detects congestion, speeding back up gradually. this prevents it from overwhelming the network.

because of all this overhead, TCP is slower but reliable. used whenever you can't afford to lose data — loading a webpage, sending an email, SSH.


**TCP flags**

every TCP packet carries flags in its header that indicate what type of packet it is:

```
SYN   — initiating a connection
ACK   — acknowledging received data
FIN   — closing a connection gracefully
RST   — resetting/forcibly closing a connection
PSH   — push data to application immediately
URG   — urgent data
```

these flags are extremely useful for anomaly detection:
- **SYN with no ACK follow-up** — connection was never completed. a flood of these = SYN flood DDoS attack
- **RST flood** — many forcible resets = something is actively rejecting connections
- **SYN to many different ports** — port scanning, someone probing what services are open
- **FIN/RST with no prior SYN** — packet is part of no known connection, possibly spoofed or a scan


### UDP

UDP is connectionless — no handshake, no setup. you just send packets and hope they arrive. there's no acknowledgement, no retransmission, no ordering.

```
Client  →  data  →  Server   (no handshake, just sends)
```

this makes UDP much faster and lower latency. used when:
- **speed matters more than perfection** — video streaming, online gaming. a dropped frame is better than a frozen screen waiting for retransmission
- **the app handles reliability itself** — e.g. QUIC (used by modern HTTP/3) is built on UDP but adds its own reliability layer
- **tiny one-off requests** — DNS queries are a single small request/response. the overhead of a TCP handshake would double the time for no benefit

### TCP vs UDP

|             | TCP                                           | UDP                          |
| ----------- | --------------------------------------------- | ---------------------------- |
| Connection  | yes, handshake required                       | no, just send                |
| Reliability | guaranteed delivery, retransmits lost packets | no guarantees                |
| Order       | in-order delivery                             | may arrive out of order      |
| Speed       | slower (overhead)                             | faster (no overhead)         |
| Use cases   | HTTP, SSH, email                              | DNS, video streaming, gaming |

### MTU

maximum transmission unit — the largest packet a link can carry without fragmentation. standard Ethernet = **1500 bytes**. data centres often use **jumbo frames = 9000 bytes** to reduce overhead.

if a packet is too large it gets fragmented (split into smaller pieces) or the sender is told to reduce size via Path MTU Discovery (PMTUD).

a MTU mismatch between links causes **silent packet drops** — packets disappear with no obvious error. a common real-world debugging headache.