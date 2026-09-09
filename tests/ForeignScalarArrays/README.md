# Native scalar reference projections

Run `python3 tests/ForeignScalarArrays/run.py src/bin/Debug/net10.0/Composer`
from the Composer checkout. The driver retains fresh binaries, logs and
compiler intermediates, uses timeouts, and disables core dumps for the expected
rejection. Both gates passed on Composer Debug build 33 on 2026-09-09; local
evidence is `/tmp/clef-scalar-arrays-7ojuqzvb`.

These are actual libc calls from compiled Clef. Their declarations follow the
installed `unistd.h`, `arpa/inet.h` and `wchar.h` on the explicitly selected
Linux x86_64 platform. There is no C implementation or host-language substitute.
An unrelated array containing 4294967296 forces wider ordinary integer storage.

- `pipe` writes two native signed 32-bit file descriptors.
- `write` reads all eight declared octets, including 128 and 255. Its original
  source and aliases remain unchanged; retained IR verifies no copyback.
- `read` writes those octets back into the wider source array. Existing aliases
  see the values, and the two untouched trailing elements remain intact.
- `inet_pton` uses a declared one-word IPv4 destination policy. The value
  0xffffffff becomes 4294967295 after unsigned 32-bit copyback.
- `wmemset` writes the signed 32-bit minimum into the first element, preserving
  the tail and sign-extending the result into wider storage. Its returned
  temporary-buffer alias is immediately ignored; this does not establish a
  usable escaping handle to projected storage.
- A separate executable gives `write` a final element of 256. The range guard
  terminates it before any bytes reach stdout.

The positive case also passes with allocator storage perturbed to 165. The
adapter’s contract here is synchronous: native code does not retain projected
arrays after returning. This gate does not establish multi-array aliasing or
asynchronous retention support.
