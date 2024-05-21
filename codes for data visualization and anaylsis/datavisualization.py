import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# List of file names
file_names = ['sit.csv', 'std.csv', 'wlk.csv', 'ups4.csv', 'dws2.csv','backhand.csv', 'forehand2.csv', 'serve2.csv']
num_rows = 1500  # Change this to the desired number of rows

# Loop through each file
for file_name in file_names:
    # Step 1: Read CSV file into a DataFrame, reading only a certain number of rows
    df = pd.read_csv(file_name, nrows=num_rows)

    # Step 2: Extract relevant columns
    time = df['Time']
    accel_x = df['Accel X']
    accel_y = df['Accel Y']
    accel_z = df['Accel Z']
    gyro_x = df['Gyro X']
    gyro_y = df['Gyro Y']
    gyro_z = df['Gyro Z']

    # Step 3: Optionally apply moving average window
    window_size = 20
    smoothed_accel_x = np.convolve(accel_x, np.ones(window_size) / window_size, mode='valid')
    smoothed_accel_y = np.convolve(accel_y, np.ones(window_size) / window_size, mode='valid')
    smoothed_accel_z = np.convolve(accel_z, np.ones(window_size) / window_size, mode='valid')
    smoothed_gyro_x = np.convolve(gyro_x, np.ones(window_size) / window_size, mode='valid')
    smoothed_gyro_y = np.convolve(gyro_y, np.ones(window_size) / window_size, mode='valid')
    smoothed_gyro_z = np.convolve(gyro_z, np.ones(window_size) / window_size, mode='valid')

    # Step 4: Plot the data
    plt.figure(figsize=(14, 10))  # Larger figsize

    plt.subplot(2, 1, 1)
    plt.plot(time, accel_x, label='Accel X')
    plt.plot(time, accel_y, label='Accel Y')
    plt.plot(time, accel_z, label='Accel Z')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.title(f'Accelerometer Data ({file_name})')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time, gyro_x, label='Gyro X')
    plt.plot(time, gyro_y, label='Gyro Y')
    plt.plot(time, gyro_z, label='Gyro Z')
    plt.xlabel('Time')
    plt.ylabel('Angular Velocity')
    plt.title(f'Gyroscope Data ({file_name})')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot the smoothed data
    plt.figure(figsize=(14, 10))  # Larger figsize

    plt.subplot(2, 1, 1)
    plt.plot(time[window_size - 1:], smoothed_accel_x, linestyle='--', label='Smoothed Accel X')
    plt.plot(time[window_size - 1:], smoothed_accel_y, linestyle='--', label='Smoothed Accel Y')
    plt.plot(time[window_size - 1:], smoothed_accel_z, linestyle='--', label='Smoothed Accel Z')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.title(f'Smoothed Accelerometer Data ({file_name})')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time[window_size - 1:], smoothed_gyro_x, linestyle='--', color='blue', label='Smoothed Gyro X')
    plt.plot(time[window_size - 1:], smoothed_gyro_y, linestyle='--', color='orange', label='Smoothed Gyro Y')
    plt.plot(time[window_size - 1:], smoothed_gyro_z, linestyle='--', color='green', label='Smoothed Gyro Z')
    plt.xlabel('Time')
    plt.ylabel('Angular Velocity')
    plt.title(f'Smoothed Gyroscope Data ({file_name})')
    plt.legend()

    plt.tight_layout()
    plt.show()
