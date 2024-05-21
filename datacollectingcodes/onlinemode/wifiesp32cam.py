import socket
import csv

# Define server parameters
HOST = '0.0.0.0'  # Listen on all available interfaces
PORT = 12345       # Use the same port as in your ESP32 code

# Specify the path to the CSV file
CSV_FILE_PATH = r"C:\Users\Dell\Desktop\sensor_data.csv"

# Create a TCP/IP socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    # Bind the socket to the address and port
    server_socket.bind((HOST, PORT))

    # Listen for incoming connections
    server_socket.listen()

    print(f"Server listening on {HOST}:{PORT}")

    # Accept a client connection
    client_socket, client_address = server_socket.accept()
    print(f"Connection from {client_address}")

    # Create a CSV file to store the received data
    with open(CSV_FILE_PATH, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Timestamp', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz'])  # Write header

        while True:
            # Receive data from the client
            data = client_socket.recv(1024).decode().strip()
            if not data:
                print("Client disconnected.")
                break

            # Split the received data into individual entries
            entries = data.split('\r\n')
            for entry in entries:
                if entry:  # Ensure entry is not empty
                    # Split the entry into values
                    values = entry.split(',')
                    if len(values) == 7:  # Ensure the entry contains all expected values
                        try:
                            # Extract the values
                            timestamp, ax, ay, az, gx, gy, gz = map(int, values)
                            # Write the received data to the CSV file
                            csv_writer.writerow([timestamp, ax, ay, az, gx, gy, gz])
                            print("Data received and stored:", timestamp, ax, ay, az, gx, gy, gz)
                        except ValueError:
                            print("Invalid entry:", entry)
                    else:
                        print("Invalid entry:", entry)

# Server has finished
print("Server closed.")
