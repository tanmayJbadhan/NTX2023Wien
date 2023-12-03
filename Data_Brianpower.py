import socket
import csv

UDP_IP = "127.0.0.1"
UDP_PORT = 1000

def parse_eeg_data(data):
    values = data.decode().split(',')
    parsed_values = [float(value) if value != 'NaN' else None for value in values]
    return parsed_values

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# Open a CSV file to write the data
with open('eeg_data_blink.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # Define your CSV headers based on the EEG data structure
    headers = [
        'Delta1', 'Delta2', 'Delta3', 'Delta4', 'Delta5', 'Delta6', 'Delta7', 'Delta8',
        'Theta1', 'Theta2', 'Theta3', 'Theta4', 'Theta5', 'Theta6', 'Theta7', 'Theta8',
        'Alpha1', 'Alpha2', 'Alpha3', 'Alpha4', 'Alpha5', 'Alpha6', 'Alpha7', 'Alpha8',
        'BetaLow1', 'BetaLow2', 'BetaLow3', 'BetaLow4', 'BetaLow5', 'BetaLow6', 'BetaLow7', 'BetaLow8',
        'BetaMid1', 'BetaMid2', 'BetaMid3', 'BetaMid4', 'BetaMid5', 'BetaMid6', 'BetaMid7', 'BetaMid8',
        'BetaHigh1', 'BetaHigh2', 'BetaHigh3', 'BetaHigh4', 'BetaHigh5', 'BetaHigh6', 'BetaHigh7', 'BetaHigh8',
        'Gamma1', 'Gamma2', 'Gamma3', 'Gamma4', 'Gamma5', 'Gamma6', 'Gamma7', 'Gamma8',
        'Delta_Avg', 'Theta_Avg', 'Alpha_Avg', 'BetaLow_Avg', 'BetaMid_Avg', 'BetaHigh_Avg', 'Gamma_Avg',
        'Delta_Bipolar_Avg', 'Theta_Bipolar_Avg', 'Alpha_Bipolar_Avg', 'BetaLow_Bipolar_Avg', 'BetaMid_Bipolar_Avg', 'BetaHigh_Bipolar_Avg', 'Gamma_Bipolar_Avg',
        'Label'
    ]
    writer.writerow(headers)

    while True:
        data, addr = sock.recvfrom(1024)  # buffer size is 1024 bytes
        print("received message:", data)
        
        # Parse the data
        parsed_data = parse_eeg_data(data)

        # Append the label 'Crunch'
        parsed_data.append('Crunch')

        # Write to CSV
        writer.writerow(parsed_data)
