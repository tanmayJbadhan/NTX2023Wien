import UnicornPy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, lfilter


def main():
    # Specifications for the data acquisition.
    #-------------------------------------------------------------------------------------
    TestsignaleEnabled = False;
    FrameLength = 1;
    AcquisitionDurationInSeconds = 10;
    DataFile = "data.csv";
    
    print("Unicorn Acquisition Example")
    print("---------------------------")
    print()

    # Setup for real-time plotting
    plt.ion()  # Turn on interactive mode for real-time plotting
    fig, ax = plt.subplots()  # Create a figure and an axes
    line, = ax.plot([], [], lw=2)  # Initialize a line object for updating
    ax.set_ylim([-100, 100])  # Set Y-axis limits
    try:
        # Get available devices.
        #-------------------------------------------------------------------------------------
        
         # Notch filter setup
        fs = UnicornPy.SamplingRate  # Sampling rate
        f0_1 = 50  # Frequency to be removed (50Hz)
        f0_2 = 60  # Frequency to be removed (60Hz)
        Q = 30  # Quality factor

        # Create notch filters
        b1, a1 = iirnotch(f0_1, Q, fs)
        b2, a2 = iirnotch(f0_2, Q, fs)
        # Get available device serials.
        deviceList = UnicornPy.GetAvailableDevices(True)

        if len(deviceList) <= 0 or deviceList is None:
            raise Exception("No device available.Please pair with a Unicorn first.")

        # Print available device serials.
        print("Available devices:")
        i = 0
        for device in deviceList:
            print("#%i %s" % (i,device))
            i+=1

        # Request device selection.
        print()
        deviceID = int(input("Select device by ID #"))
        if deviceID < 0 or deviceID > len(deviceList):
            raise IndexError('The selected device ID is not valid.')

        # Open selected device.
        #-------------------------------------------------------------------------------------
        print()
        print("Trying to connect to '%s'." %deviceList[deviceID])
        device = UnicornPy.Unicorn(deviceList[deviceID])
        print("Connected to '%s'." %deviceList[deviceID])
        print()

        # Create a file to store data.
        file = open(DataFile, "wb")

        # Initialize acquisition members.
        #-------------------------------------------------------------------------------------
        numberOfAcquiredChannels= device.GetNumberOfAcquiredChannels()
        configuration = device.GetConfiguration()

        # Print acquisition configuration
        print("Acquisition Configuration:");
        print("Sampling Rate: %i Hz" %UnicornPy.SamplingRate);
        print("Frame Length: %i" %FrameLength);
        print("Number Of Acquired Channels: %i" %numberOfAcquiredChannels);
        print("Data Acquisition Length: %i s" %AcquisitionDurationInSeconds);
        print();

        # Allocate memory for the acquisition buffer.
        receiveBufferBufferLength = FrameLength * numberOfAcquiredChannels * 4
        receiveBuffer = bytearray(receiveBufferBufferLength)

        try:
            # Start data acquisition.
            #-------------------------------------------------------------------------------------
            device.StartAcquisition(TestsignaleEnabled)
            print("Data acquisition started.")

            # Calculate number of get data calls.
            numberOfGetDataCalls = int(AcquisitionDurationInSeconds * UnicornPy.SamplingRate / FrameLength);
        
            # Limit console update rate to max. 25Hz or slower to prevent acquisition timing issues.                   
            consoleUpdateRate = int((UnicornPy.SamplingRate / FrameLength) / 25.0);
            if consoleUpdateRate == 0:
                consoleUpdateRate = 1

            # Acquisition loop.
            #-------------------------------------------------------------------------------------
            for i in range (0,numberOfGetDataCalls):
                # Receives the configured number of samples from the Unicorn device and writes it to the acquisition buffer.
                device.GetData(FrameLength,receiveBuffer,receiveBufferBufferLength)

                # Convert receive buffer to numpy float array 
                data = np.frombuffer(receiveBuffer, dtype=np.float32, count=numberOfAcquiredChannels * FrameLength)
                data = np.reshape(data, (FrameLength, numberOfAcquiredChannels))
                np.savetxt(file,data,delimiter=',',fmt='%.3f',newline='\n')
                
                # Apply notch filters
                data_filtered = lfilter(b1, a1, data, axis=0)
                data_filtered = lfilter(b2, a2, data_filtered, axis=0)

                # Real-time data plotting
                line.set_xdata(np.arange(data_filtered.shape[0]))
                line.set_ydata(data_filtered[:, 0])  # Assuming we plot the first channel
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
                # Update console to indicate that the data acquisition is running.
                if i % consoleUpdateRate == 0:
                    print('.',end='',flush=True)

            # Stop data acquisition.
            #-------------------------------------------------------------------------------------
            device.StopAcquisition();
            print()
            print("Data acquisition stopped.");

        except UnicornPy.DeviceException as e:
            print(e)
        except Exception as e:
            print("An unknown error occured. %s" %e)
        finally:
            # release receive allocated memory of receive buffer
            del receiveBuffer

            #close file
            file.close()

            # Close device.
            #-------------------------------------------------------------------------------------
            del device
            print("Disconnected from Unicorn")

    except Unicorn.DeviceException as e:
        print(e)
    except Exception as e:
        print("An unknown error occured. %s" %e)

    input("\n\nPress ENTER key to exit")

#execute main
main()
