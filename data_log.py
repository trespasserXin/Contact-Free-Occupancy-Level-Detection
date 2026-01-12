# import time
# import csv
# from threading import Thread, Event
# from queue import Queue, Empty
#
# import serial
# import sounddevice as sd
# import soundfile as sf
# import numpy as np
# from serial import SerialException
#
# # ===================== CONFIG =====================
# SERIAL_PORT = "COM3"
# BAUDRATE = 115200
# SERIAL_TIMEOUT = 1.0  # seconds
#
# AUDIO_FILE = "mic.wav"
# SENSOR_LOG_FILE = "log.csv"
#
# SAMPLERATE = 48000
# CHANNELS = 2  # 1 = mono, 2 = stereo
#
# # ===================== GLOBALS =====================
# serial_queue = Queue()
# stop_event = Event()  # used to tell threads/callbacks to stop
#
# # ===================== SERIAL READER =====================
# def serial_reader():
#     """Open COM port once and stream lines into a queue."""
#     try:
#         ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=SERIAL_TIMEOUT)
#         print(f"[Serial] Opened {SERIAL_PORT} at {BAUDRATE} baud")
#     except SerialException as e:
#         print(f"[Serial] Failed to open {SERIAL_PORT}: {e}")
#         return
#
#     try:
#         while not stop_event.is_set():
#             try:
#                 line = ser.readline().decode(errors="ignore").strip()
#                 if line:
#                     print("line here")
#                     ts_ms = int(time.time() * 1000)
#                     serial_queue.put((ts_ms, line))
#             except SerialException as e:
#                 print(f"[Serial] Error while reading: {e}")
#                 break
#     finally:
#         ser.close()
#         print("[Serial] Port closed")
#
# # ===================== SERIAL LOGGER =====================
# def serial_logger():
#     """Write serial data from queue into CSV."""
#     with open(SENSOR_LOG_FILE, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         # You can store when audio started later if you want to sync precisely
#         writer.writerow(["timestamp_ms", "sensor_value"])
#         print(f"[Logger] Logging serial to {SENSOR_LOG_FILE}")
#
#         while not stop_event.is_set():
#             try:
#                 ts_ms, line = serial_queue.get(timeout=0.1)
#                 writer.writerow([ts_ms, line])
#                 # Optional: flush occasionally if you're paranoid about crashes
#                 # f.flush()
#                 print(f"[Serial] {ts_ms} {line}")
#             except Empty:
#                 # No new serial data, just loop again
#                 # print("[Serial] No new data")
#                 continue
#
#
#         # Drain any remaining items in the queue before exiting
#         while True:
#             try:
#                 ts_ms, line = serial_queue.get_nowait()
#                 writer.writerow([ts_ms, line])
#             except Empty:
#                 break
#
#         print("[Logger] Serial logger stopped")
#
# # ===================== AUDIO RECORDING =====================
# def record_audio():
#     """Record microphone to WAV file until stop_event is set."""
#     print(f"[Audio] Writing raw audio to {AUDIO_FILE}")
#     audio_start_time = time.time()
#     print(f"[Audio] Start time (epoch s): {audio_start_time}")
#
#     # Open sound file for streaming writes
#     with sf.SoundFile(
#         AUDIO_FILE,
#         mode="w",
#         samplerate=SAMPLERATE,
#         channels=CHANNELS,
#         subtype="PCM_16",  # 16-bit PCM, good for most work
#     ) as wf:
#
#         def audio_callback(indata, frames, time_info, status):
#             if status:
#                 print(f"[Audio] Status: {status}", flush=True)
#
#             # indata shape: (frames, channels), dtype float32
#             # Convert to something appropriate if needed; here we just write as is.
#             wf.write(indata)
#
#             # Stop if main thread requested
#             if stop_event.is_set():
#                 raise sd.CallbackStop()
#
#         # Audio stream
#         with sd.InputStream(
#             channels=CHANNELS,
#             samplerate=SAMPLERATE,
#             callback=audio_callback,
#             blocksize=0,  # let sounddevice choose
#         ):
#             print("[Audio] Recording... press Ctrl+C to stop.")
#             while not stop_event.is_set():
#                 time.sleep(0.01)
#
#     print("[Audio] Recording stopped, file closed")
#
# # ===================== MAIN =====================
# def main():
#     # Start serial threads
#     t_serial = Thread(target=serial_reader, daemon=True)
#     t_logger = Thread(target=serial_logger, daemon=True)
#
#     t_serial.start()
#     t_logger.start()
#
#     try:
#         # Run audio recording in main thread (so Ctrl+C works nicely)
#         record_audio()
#     except KeyboardInterrupt:
#         print("\n[Main] Ctrl+C received, stopping...")
#     finally:
#         stop_event.set()
#         # Give threads a moment to finish
#         t_serial.join(timeout=2)
#         t_logger.join(timeout=2)
#         print("[Main] All done")
#
# if __name__ == "__main__":
#     main()

import time
import csv
from threading import Thread, Event
from queue import Queue, Empty

import serial
import sounddevice as sd
import soundfile as sf
import numpy as np
import struct
from serial import SerialException

# ===================== CONFIG =====================
SERIAL_PORT = "COM3"
BAUDRATE = 115200
SERIAL_TIMEOUT = 1.0  # seconds

AUDIO_FILE = "mic.wav"
MIC_LOG_FILE = "mic_raw.bin"
MIC_STRUCT = struct.Struct("<Ih")
SENSOR_LOG_FILE = "log.csv"

SAMPLERATE = 48000
CHANNELS = 2  # 1 = mono, 2 = stereo

# ===================== GLOBALS =====================
serial_queue = Queue()
audio_queue = Queue()
stop_event = Event()  # used to tell threads/callbacks to stop

# ===================== SERIAL READER =====================
def serial_reader():
    """Open COM port once and stream lines into a queue."""
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=SERIAL_TIMEOUT)
        print(f"[Serial] Opened {SERIAL_PORT} at {BAUDRATE} baud")
    except SerialException as e:
        print(f"[Serial] Failed to open {SERIAL_PORT}: {e}")
        return

    try:
        while not stop_event.is_set():
            try:
                line = ser.readline().decode(errors="ignore").strip()
                if line:
                    ts_ms = int(time.time())
                    serial_queue.put((ts_ms, line))
            except SerialException as e:
                print(f"[Serial] Error while reading: {e}")
                break
    finally:
        ser.close()
        print("[Serial] Port closed")

# ===================== SERIAL LOGGER =====================
def serial_logger():
    """Write serial data from queue into CSV."""
    NO_DATA_TIMEOUT = 5.0
    with open (MIC_LOG_FILE, "wb") as mic_f, \
         open(SENSOR_LOG_FILE, "w", newline="") as serial_file:

        serial_writer = csv.writer(serial_file)

        # mic_writer.writerow(['t_ms', 'mic_raw'])
        serial_writer.writerow(['t_ms', 'Temp', 'RH', 'Gas', 'Xg','Yg','Zg','vLight','Light',
                              'USDistance', 'Corrected Distance'])

        print(f"[Logger] Logging serial to {SENSOR_LOG_FILE} and {MIC_LOG_FILE}")

        start_time = time.time()
        got_any_data = False
        warned_no_data = False

        while not stop_event.is_set():
            try:
                ts_ms, line = serial_queue.get(timeout=0.1)
                # print(line)
                content = line.split(',')
                kind = content[0]
                if kind == 'MIC':
                    if len(content) < 3:
                        continue  # malformed line
                    try:
                        t_ms = ts_ms
                        raw = int(content[2])
                    except ValueError:
                        continue
                    mic_f.write(MIC_STRUCT.pack(t_ms, raw))
                    # ---------- SENS line ----------
                    # Format:
                    # SENS,t_ms,T,RH,G,Xg,Yg,Zg,vLight,Light_raw,US,US_corrected
                elif kind == "SENS":
                    if len(content) < 12:
                        continue  # malformed line

                    try:
                        t_ms = ts_ms
                        T_C = float(content[2])
                        RH = float(content[3])
                        Gas_kOhm = float(content[4])
                        Xg = float(content[5])
                        Yg = float(content[6])
                        Zg = float(content[7])
                        vLight_V = float(content[8])
                        Light_raw = int(content[9])
                        US_mm = float(content[10])
                        US_corr_mm = float(content[11])
                    except ValueError:
                        continue  # bad parse, skip

                    serial_writer.writerow([
                        t_ms,
                        T_C,
                        RH,
                        Gas_kOhm,
                        Xg,
                        Yg,
                        Zg,
                        vLight_V,
                        Light_raw,
                        US_mm,
                        US_corr_mm,
                    ])

                # Mark that we have received at least one line
                if not got_any_data:
                    got_any_data = True
                # Optional: flush sometimes if you care about crashes
                # f.flush()
                # print(f"[Serial] {ts_ms} {line}")  # keep or comment for less spam
            except Empty:
                # Just no data at this moment; don't spam prints
                if (not got_any_data
                        and not warned_no_data
                        and (time.time() - start_time) > NO_DATA_TIMEOUT):
                    print("[Logger] WARNING: No serial data received yet. "
                          "Check COM port / wiring / baudrate.")
                    warned_no_data = True
                continue

        # Drain remaining items before exiting
        while True:
            try:
                ts_ms, line = serial_queue.get_nowait()
                # writer.writerow([ts_ms, line])
                print(f"Remaining [Serial] {ts_ms} {line}")
            except Empty:
                break

        print("[Logger] Serial logger stopped")

# ===================== AUDIO WRITER =====================
def audio_writer():
    """Write audio blocks from queue into WAV file."""
    print(f"[Audio] Writing raw audio to {AUDIO_FILE}")
    audio_start_time = time.time()
    print(f"[Audio] Start time (epoch s): {audio_start_time}")

    with sf.SoundFile(
        AUDIO_FILE,
        mode="w",
        samplerate=SAMPLERATE,
        channels=CHANNELS,
        subtype="PCM_16",
    ) as wf:
        while not stop_event.is_set() or not audio_queue.empty():
            try:
                block = audio_queue.get(timeout=0.1)
                wf.write(block)
            except Empty:
                continue

    print("[Audio] WAV file closed")

# ===================== AUDIO RECORDING =====================
def record_audio():
    """Record microphone and push blocks into audio_queue until stop_event is set."""

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[Audio] Status: {status}", flush=True)

        # Just queue the data; real work happens in audio_writer thread
        try:
            audio_queue.put_nowait(indata.copy())
            time.sleep(0.01)
        except:
            # If queue is full or something goes wrong, you can drop data
            # to avoid blocking the callback.
            pass

        if stop_event.is_set():
            raise sd.CallbackStop()

    with sd.InputStream(
        channels=CHANNELS,
        samplerate=SAMPLERATE,
        callback=audio_callback,
        blocksize=0,  # let sounddevice choose
    ):
        print("[Audio] Recording... press Ctrl+C to stop.")
        while not stop_event.is_set():
            time.sleep(0.01)

    print("[Audio] Recording stopped")

# ===================== MAIN =====================
def main():
    # Start serial threads
    t_serial = Thread(target=serial_reader, daemon=True)
    t_logger = Thread(target=serial_logger, daemon=True)
    t_audio_writer = Thread(target=audio_writer, daemon=True)

    t_serial.start()
    t_logger.start()
    t_audio_writer.start()

    try:
        # Run audio recording in main thread (so Ctrl+C works nicely)
        record_audio()
    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C received, stopping...")
    finally:
        stop_event.set()
        # Give threads a moment to finish
        t_serial.join(timeout=2)
        t_logger.join(timeout=2)
        t_audio_writer.join(timeout=2)
        print("[Main] All done")

if __name__ == "__main__":
    main()
