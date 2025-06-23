# === Import Required Libraries ===
import serial                                     # pySerial: for serial communication
import time                                       # time: (not used here, but useful for timing control)
from datetime import datetime                     # datetime: to timestamp the output file


def main():
    serial_port = 'COM3'                          # Serial port name (change as needed for your system)
    baud_rate = 115200                             # Baud rate (must match sender device)

    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)  # Open serial port with 1 sec timeout
        print(f"Connected to {serial_port} at {baud_rate} baud.")
    except serial.SerialException:
        print(f"Failed to connect to {serial_port}")             # If port doesn't exist or is busy
        return

    # Create a filename with the current timestamp (e.g., accel_data_20250623_142530.csv)
    filename = f"accel_data/accel_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with open(filename, 'w') as f:
        print(f"Logging data to {filename}")
        try:
            while True:                                           # Infinite loop to read incoming data
                line = ser.readline().decode('utf-8', errors='ignore').strip()  # Read and decode line
                if line:                                          # Ignore empty lines
                    if line.startswith("CSV_HEADER"):
                        # If the line contains a custom header (like "CSV_HEADER,Timestamp(ms),X,Y,Z")
                        header = line.split(",", 1)[1] if "," in line else "Timestamp(ms),X,Y,Z"
                        f.write(header + "\n")                    # Write header to file
                        print(f"Header: {header}")
                    else:
                        f.write(line + "\n")                      # Write data line to file
                        print(line)                               # Print to console for live monitoring
        except KeyboardInterrupt:
            print("\nLogging stopped by user.")                   #  Stop the loop when Ctrl+C is pressed
        finally:
            ser.close()                                           # Always close the port when done
            print("Serial port closed.")

if __name__ == "__main__":
    main()                                                        # Start the script
