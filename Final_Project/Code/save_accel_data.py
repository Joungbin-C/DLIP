import serial
import time
from datetime import datetime

def main():
    serial_port = 'COM3'
    baud_rate = 115200

    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        print(f"Connected to {serial_port} at {baud_rate} baud.")
    except serial.SerialException:
        print(f"Failed to connect to {serial_port}")
        return

    filename = f"accel_data/accel_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with open(filename, 'w') as f:
        print(f"Logging data to {filename}")
        try:
            while True:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    if line.startswith("CSV_HEADER"):
                        header = line.split(",", 1)[1] if "," in line else "Timestamp(ms),X,Y,Z"
                        f.write(header + "\n")
                        print(f"Header: {header}")
                    else:
                        f.write(line + "\n")
                        print(line)
        except KeyboardInterrupt:
            print("\nLogging stopped by user.")
        finally:
            ser.close()
            print("Serial port closed.")

if __name__ == "__main__":
    main()
