"""
Simultaneous BLE RSSI + USRP IQ capture for DL training data.

Runs BLE binary RSSI capture and USRP IQ capture in parallel threads,
saving both to a single .npz file for time-aligned dataset building.

Requirements:
  pip install pyserial uhd numpy

Usage:
  python capture_simultaneous.py --ble-port COM5 --duration 45 --out capture_001.npz
  python capture_simultaneous.py --ble-port COM5 --ble-direct --usrp-gain 30 --duration 60 --scenario breathing_1m
"""

import argparse
import struct
import time
import threading
import numpy as np

# Binary packet protocol (shared with capture_rssi_stream.py)
MAGIC = 0xDEADBEEF
MAGIC_LE = struct.pack("<I", MAGIC)
HDR_SIZE = 16
DATA_SIZE = 1024
PKT_SIZE = HDR_SIZE + DATA_SIZE
HDR_FMT = "<IIIHBB"

BLE_RATE = 77000  # nRF54L15: ~77 kHz inline stream (measured actual rate)


def ble_capture_thread(port, duration, direct, result):
    """Capture BLE RSSI binary stream in a thread."""
    import serial

    baud_binary = 921600

    if direct:
        ser = serial.Serial(port, baud_binary, timeout=1)
        time.sleep(0.3)
        ser.reset_input_buffer()
    else:
        ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(0.5)
        ser.reset_input_buffer()

        # Wait for SWITCHING_BAUD
        t0 = time.time()
        while (time.time() - t0) < 60:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if line:
                print(f"  [BLE] {line}")
            if line.startswith("SWITCHING_BAUD"):
                parts = line.split()
                if len(parts) >= 2:
                    baud_binary = int(parts[1])
                break
        else:
            result['error'] = "Never received SWITCHING_BAUD"
            ser.close()
            return

        time.sleep(0.3)
        ser.baudrate = baud_binary
        ser.reset_input_buffer()
        time.sleep(0.2)

    # Signal ready
    result['ready'] = True
    result['wall_time_start'] = time.time()

    # Wait for global start signal
    while not result.get('go', False):
        time.sleep(0.001)

    # Capture packets
    packets_data = []
    total_samples = 0
    total_drops = 0
    last_seq = -1
    seq_gaps = 0
    t_start = time.time()
    rxbuf = b''

    try:
        while (time.time() - t_start) < duration:
            chunk = ser.read(max(PKT_SIZE, ser.in_waiting or PKT_SIZE))
            if not chunk:
                continue
            rxbuf += chunk

            while len(rxbuf) >= PKT_SIZE:
                idx = rxbuf.find(MAGIC_LE)
                if idx < 0:
                    rxbuf = rxbuf[-3:]
                    break
                if idx > 0:
                    rxbuf = rxbuf[idx:]
                if len(rxbuf) < PKT_SIZE:
                    break

                magic, seq, start_sample, drop_count, buf_id, reserved = \
                    struct.unpack_from(HDR_FMT, rxbuf, 0)

                rssi_data = rxbuf[HDR_SIZE:HDR_SIZE + DATA_SIZE]
                rxbuf = rxbuf[PKT_SIZE:]

                # Store raw bytes
                packets_data.append(bytes(rssi_data))
                total_samples += DATA_SIZE
                total_drops = drop_count

                if last_seq >= 0 and seq != last_seq + 1 and len(packets_data) > 2:
                    seq_gaps += 1
                last_seq = seq

    except Exception as e:
        result['error'] = str(e)

    ser.close()
    elapsed = time.time() - t_start

    # Convert to numpy array
    if packets_data:
        all_rssi = np.concatenate([
            np.frombuffer(d, dtype=np.int8).astype(np.float64)
            for d in packets_data
        ])
    else:
        all_rssi = np.array([])

    result['rssi'] = all_rssi
    result['rate'] = BLE_RATE
    result['n_packets'] = len(packets_data)
    result['drops'] = total_drops
    result['seq_gaps'] = seq_gaps
    result['elapsed'] = elapsed

    print(f"  [BLE] Done: {len(packets_data)} pkts, {total_samples} samples, "
          f"drops={total_drops}, gaps={seq_gaps}")


def usrp_capture_thread(freq, rate, gain, duration, result):
    """Capture USRP IQ in a thread."""
    try:
        import uhd
    except ImportError:
        result['error'] = "uhd not installed"
        result['ready'] = True
        return

    usrp = uhd.usrp.MultiUSRP()
    usrp.set_rx_rate(rate, 0)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(freq), 0)
    usrp.set_rx_gain(gain, 0)
    usrp.set_rx_antenna("RX2", 0)

    while not usrp.get_rx_sensor("lo_locked", 0).to_bool():
        time.sleep(0.01)

    actual_rate = usrp.get_rx_rate(0)
    actual_freq = usrp.get_rx_freq(0)
    actual_gain = usrp.get_rx_gain(0)
    print(f"  [USRP] Configured: freq={actual_freq/1e9:.6f} GHz, "
          f"rate={actual_rate/1e6:.3f} MHz, gain={actual_gain:.1f} dB")

    n_samples = int(duration * actual_rate)
    iq_data = np.empty(n_samples, dtype=np.complex64)

    # Setup streaming
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    streamer = usrp.get_rx_stream(st_args)
    metadata = uhd.types.RXMetadata()
    buffer_size = min(int(actual_rate), 100000)

    # Signal ready
    result['ready'] = True

    # Wait for global start signal
    while not result.get('go', False):
        time.sleep(0.001)

    # Start streaming
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    streamer.issue_stream_cmd(stream_cmd)

    result['wall_time_start'] = time.time()

    idx = 0
    overflows = 0
    while idx < n_samples:
        remaining = n_samples - idx
        chunk = min(buffer_size, remaining)
        buf = np.empty(chunk, dtype=np.complex64)
        n_recv = streamer.recv(buf, metadata)

        if metadata.error_code == uhd.types.RXMetadataErrorCode.none:
            iq_data[idx:idx + n_recv] = buf[:n_recv]
            idx += n_recv
        elif metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
            overflows += 1
        elif metadata.error_code == uhd.types.RXMetadataErrorCode.timeout:
            continue
        else:
            break

    # Stop
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    streamer.issue_stream_cmd(stream_cmd)

    result['iq'] = iq_data[:idx]
    result['rate'] = actual_rate
    result['freq'] = actual_freq
    result['gain'] = actual_gain
    result['overflows'] = overflows

    print(f"  [USRP] Done: {idx} samples ({idx/actual_rate:.2f}s), "
          f"overflows={overflows}")


def main():
    parser = argparse.ArgumentParser(
        description="Simultaneous BLE RSSI + USRP IQ capture"
    )
    # BLE args
    parser.add_argument("--ble-port", required=True,
                        help="BLE serial port (e.g., COM5)")
    parser.add_argument("--ble-direct", action="store_true",
                        help="BLE direct mode (already streaming at 921600)")

    # USRP args
    parser.add_argument("--usrp-freq", type=float, default=2.44e9,
                        help="USRP center frequency (Hz)")
    parser.add_argument("--usrp-rate", type=float, default=2e6,
                        help="USRP sample rate (Hz)")
    parser.add_argument("--usrp-gain", type=float, default=30,
                        help="USRP RX gain (dB)")
    parser.add_argument("--no-usrp", action="store_true",
                        help="Skip USRP (BLE-only capture for testing)")

    # Common args
    parser.add_argument("--duration", type=float, default=45,
                        help="Capture duration (seconds)")
    parser.add_argument("--scenario", default="unknown",
                        help="Scenario label (e.g., breathing_1m)")
    parser.add_argument("--out", default="capture.npz",
                        help="Output .npz file")
    args = parser.parse_args()

    print(f"=== Simultaneous Capture ===")
    print(f"  Duration: {args.duration}s, Scenario: {args.scenario}")
    print(f"  BLE: {args.ble_port} {'(direct)' if args.ble_direct else ''}")
    if not args.no_usrp:
        print(f"  USRP: {args.usrp_freq/1e9:.4f} GHz, "
              f"{args.usrp_rate/1e6:.1f} MHz, gain={args.usrp_gain} dB")

    # Shared state
    ble_result = {'ready': False, 'go': False}
    usrp_result = {'ready': False, 'go': False}

    # Start threads
    t_ble = threading.Thread(target=ble_capture_thread,
                             args=(args.ble_port, args.duration,
                                   args.ble_direct, ble_result))
    t_ble.start()

    if not args.no_usrp:
        t_usrp = threading.Thread(target=usrp_capture_thread,
                                  args=(args.usrp_freq, args.usrp_rate,
                                        args.usrp_gain, args.duration,
                                        usrp_result))
        t_usrp.start()
    else:
        usrp_result['ready'] = True

    # Wait for both to be ready
    print("\nWaiting for devices to be ready...")
    while not ble_result.get('ready', False):
        if 'error' in ble_result:
            print(f"BLE error: {ble_result['error']}")
            return
        time.sleep(0.1)
    print("  BLE ready")

    while not usrp_result.get('ready', False):
        if 'error' in usrp_result:
            print(f"USRP error: {usrp_result['error']}")
            return
        time.sleep(0.1)
    if not args.no_usrp:
        print("  USRP ready")

    # GO!
    print(f"\n>>> Starting capture ({args.duration}s) <<<\n")
    wall_time_go = time.time()
    ble_result['go'] = True
    usrp_result['go'] = True

    # Wait for completion
    t_ble.join()
    if not args.no_usrp:
        t_usrp.join()

    # Check for errors
    if 'error' in ble_result:
        print(f"BLE capture error: {ble_result['error']}")
    if 'error' in usrp_result:
        print(f"USRP capture error: {usrp_result['error']}")

    # Save to .npz
    save_dict = {
        'ble_rssi': ble_result.get('rssi', np.array([])),
        'ble_rate': float(ble_result.get('rate', BLE_RATE)),
        'ble_wall_time': float(ble_result.get('wall_time_start', wall_time_go)),
        'ble_n_packets': int(ble_result.get('n_packets', 0)),
        'ble_drops': int(ble_result.get('drops', 0)),
        'scenario': args.scenario,
        'duration': args.duration,
        'wall_time_go': wall_time_go,
    }

    if not args.no_usrp and 'iq' in usrp_result:
        save_dict.update({
            'usrp_iq': usrp_result['iq'],
            'usrp_rate': float(usrp_result['rate']),
            'usrp_freq': float(usrp_result['freq']),
            'usrp_gain': float(usrp_result['gain']),
            'usrp_wall_time': float(usrp_result.get('wall_time_start', wall_time_go)),
            'usrp_overflows': int(usrp_result.get('overflows', 0)),
        })

    np.savez_compressed(args.out, **save_dict)

    ble_n = len(save_dict['ble_rssi'])
    usrp_n = len(save_dict.get('usrp_iq', []))
    file_size = sum(v.nbytes for v in save_dict.values()
                    if isinstance(v, np.ndarray))

    print(f"\n=== Saved {args.out} ===")
    print(f"  BLE:  {ble_n} samples ({ble_n/BLE_RATE:.1f}s)")
    if usrp_n > 0:
        print(f"  USRP: {usrp_n} samples ({usrp_n/args.usrp_rate:.1f}s)")
    print(f"  Scenario: {args.scenario}")
    print(f"  Size: {file_size/1e6:.1f} MB (uncompressed)")


if __name__ == "__main__":
    main()
