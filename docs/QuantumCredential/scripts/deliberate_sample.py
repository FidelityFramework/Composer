#!/usr/bin/env python3
"""Deliberate ADC sampling - validates each read, measures noise range.

Deploy to YoshiPi: scp deliberate_sample.py hhh@192.168.68.60:~/bin/
Run: python3 ~/bin/deliberate_sample.py [channel]
"""

import spidev
import RPi.GPIO as GPIO
import time
import sys
import math

CS_PIN = 8
spi = spidev.SpiDev()
spi.open(1, 0)
spi.max_speed_hz = 500000  # Slower: 500kHz instead of 1MHz
spi.mode = 0

GPIO.setmode(GPIO.BCM)
GPIO.setup(CS_PIN, GPIO.OUT)
GPIO.output(CS_PIN, GPIO.HIGH)

def read_adc_validated(channel=0, max_retries=3):
    """Read ADC with validation - retry if we get 0xFF or 0x00."""
    for attempt in range(max_retries):
        # Deliberate timing
        time.sleep(0.0001)  # 100µs settle time

        GPIO.output(CS_PIN, GPIO.LOW)
        time.sleep(0.00001)  # 10µs CS setup

        cmd = [0x01, (0x80 | (channel << 4)), 0x00]
        result = spi.xfer2(cmd)

        time.sleep(0.00001)  # 10µs before CS release
        GPIO.output(CS_PIN, GPIO.HIGH)

        value = ((result[1] & 0x03) << 8) | result[2]

        # Validate: reject 0x3FF (1023), 0x0FF (255), 0x000 (0)
        if value not in (0, 255, 1023):
            return value, attempt

    return None, max_retries  # Failed validation

def collect_samples(n, channel=0):
    """Collect n validated samples."""
    samples = []
    retries = 0
    failures = 0

    for i in range(n):
        value, attempts = read_adc_validated(channel)
        if value is not None:
            samples.append(value)
            retries += attempts
        else:
            failures += 1

    return samples, retries, failures

# Main
print("=" * 60)
print("Deliberate ADC Sampling - Noise Range Measurement")
print("=" * 60)

channel = 0
if len(sys.argv) > 1:
    channel = int(sys.argv[1])

print(f"\nSampling channel {channel} (500kHz SPI, validated reads)")
print("Collecting 1000 samples...\n")

start = time.time()
samples, retries, failures = collect_samples(1000, channel)
elapsed = time.time() - start

if not samples:
    print("ERROR: No valid samples collected!")
    GPIO.cleanup()
    sys.exit(1)

# Statistics
min_v = min(samples)
max_v = max(samples)
range_v = max_v - min_v
mean_v = sum(samples) / len(samples)

# Distribution in bins of 16 (for 10-bit ADC, gives 64 bins)
bins = [0] * 64
for s in samples:
    bins[s // 16] += 1

# Find the active range (bins with samples)
active_bins = [(i, c) for i, c in enumerate(bins) if c > 0]
first_bin = active_bins[0][0] if active_bins else 0
last_bin = active_bins[-1][0] if active_bins else 0

print(f"Samples collected: {len(samples)}")
print(f"Retries needed:    {retries}")
print(f"Failed reads:      {failures}")
print(f"Time elapsed:      {elapsed:.2f}s ({len(samples)/elapsed:.0f} samples/sec)")
print()
print(f"Min value:   {min_v:4d}  (0x{min_v:03X})")
print(f"Max value:   {max_v:4d}  (0x{max_v:03X})")
print(f"Range:       {range_v:4d}  counts")
print(f"Mean:        {mean_v:6.1f}")
print()

# Entropy estimation
bits_of_range = math.log2(range_v) if range_v > 0 else 0
print(f"Bits of range: {bits_of_range:.1f} (need 8.0 for full byte)")
print()

# Visual histogram (simple)
print("Distribution (each char = ~1% of samples):")
print("-" * 60)
max_count = max(bins) if bins else 1
for i in range(first_bin, last_bin + 1):
    count = bins[i]
    bar_len = int(50 * count / max_count) if max_count > 0 else 0
    low = i * 16
    high = (i + 1) * 16 - 1
    print(f"{low:4d}-{high:4d}: {'#' * bar_len} ({count})")

print("-" * 60)
print()

# LSB analysis
print("LSB Balance (for XOR viability):")
for bit in range(10):
    ones = sum(1 for s in samples if (s >> bit) & 1)
    pct = ones / len(samples) * 100
    status = "OK" if 45 <= pct <= 55 else "BIAS"
    print(f"  Bit {bit}: {pct:5.1f}% ones [{status}]")

print()
print("=" * 60)
if range_v >= 256:
    print(f"GOOD: Range of {range_v} gives {bits_of_range:.1f} bits - full byte viable")
elif range_v >= 64:
    print(f"MARGINAL: Range of {range_v} gives {bits_of_range:.1f} bits - 4-ch XOR may help")
else:
    print(f"INSUFFICIENT: Range of {range_v} gives only {bits_of_range:.1f} bits")
print("=" * 60)

GPIO.cleanup()
