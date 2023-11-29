import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer

def power_law(x, a, b):
    return a * np.power(x, b)

def detect_trend_shift(numbers):
    moving_averages = []
    
    # Calculate the moving average
    for i in range(len(numbers)):
        window = numbers[:i+1]
        moving_avg = sum(window) / len(window)
        moving_averages.append(moving_avg)
    
    # Calculate the standard deviation of the moving averages
    moving_avg_std = sum([(avg - sum(moving_averages) / len(moving_averages))**2 for avg in moving_averages]) / len(moving_averages)
    
    # Check for significant deviations from the moving average
    for i in range(len(numbers)):
        if abs(numbers[i] - moving_averages[i]) > 2 * moving_avg_std:
            ratio = numbers[i-1] / numbers[i ]
            #print(f"numbers[{i}] is {numbers[i]}   numbers[{[i-1]}] is {numbers[i - 1]} and the ration is {ratio}")
            if ratio > 2:
                return i
            
    
    return None


# Function to check if the biggest number is divisible by the smallest number and can be scaled down
def is_valid(big, small):
    return big % small == 0 and (big / small).is_integer()


#sys.argv = ["", 140.2265, 276.5712, 1.75, 1.74, 1.72, 1.68, 1.6]
arg_count = len(sys.argv)

if arg_count < 6:
    print("Usage: <small-model-IPC8> <large-model-IPC> <At least 3 MPKI; 2 for the scale models and 1 for the traget system> ")
    sys.exit(1)

small_number = int(input("Enter the smallest scale model SM (chiplet) number: "))
smallModel = small_number
#The first value is the python file and arg 2 and 3 is the IPC of scale model. Then the MPKI will be value 4 till the end.
#because it is considered as an array the 4th value will be labeled 3 in the sys.argv
#MPKI values
values = [float(arg) for arg in sys.argv[3:]]
print(len(values))

# Calculate numbers between big and small by a factor of 2X
system_SMs = [small_number]
for i in range (len(values)-1):
    small_number *= 2
    system_SMs.append(small_number)

# Print the list of numbers
print("Target system has", system_SMs[-1], " SMs (chiplets) and smallest scale model has", smallModel, "SMs (chiplets). Whole system configuration:", system_SMs)


num_of_systems = len(system_SMs)
baseline_scaleModel_cores = system_SMs[0]

##### Check if there will be a cliff ########

# Parse values from command-line arguments

trend_shift_point = detect_trend_shift(values)

cliff_expected = False
cliff_region_SM =0
superLinear_factor = 0

while True:
    
    if trend_shift_point is not None:
        #print(f"Trend shift detected at point {trend_shift_point + 2}.")
        cliff_region_SM = system_SMs[trend_shift_point]
        print(f"A cliff is expected when you move from a system with {system_SMs[trend_shift_point-1]} SMs to a system with {system_SMs[trend_shift_point]} SMs. ")
        cliff_expected = True
        while True:
            superLinear_factor = float (input("What is the fraction of time an SM in the largest scale model is unable to fetch an instruction? Value shoud be between 0 to 99. "))
            if superLinear_factor <0 or superLinear_factor > 99:
                print("Error: Value shoud be between 0 to 99.")
                continue
            break  # Exit the loop if a valid value is provided
        break
    else:
        # Do something when the answer is 'no'
        print("No cliff region is expected")
        # You can add your code here for the 'no' case
        break  # Exit the loop if the answer is 'no'
  


#Power-law
power_law_prediction = []
power_law_prediction.append(float(sys.argv[1]))
power_law_prediction.append(float(sys.argv[2]))
x = np.array([system_SMs[0], system_SMs[1]])  # Replace with your actual data
y = np.array([float(sys.argv[1]), float(sys.argv[2])]) 

params, covariance = curve_fit(power_law, x, y)
a, b = params


for i in range(num_of_systems -2):
    x_new = system_SMs[i+2]  # Replace with the desired configuration number
    predicted_performance = power_law(x_new, a, b)
    power_law_prediction.append(float(predicted_performance))




#End PowerLaw


#Linear Regression

linear_regression_prediction = []
linear_regression_prediction.append(float(sys.argv[1]))
linear_regression_prediction.append(float(sys.argv[2]))
# Sample data (replace with your data)
x = np.array([system_SMs[0], system_SMs[1]]).reshape(-1, 1)  # Configuration numbers
y = np.array([float(sys.argv[1]), float(sys.argv[2])])  # Performance numbers
model = LinearRegression()
model.fit(x, y)


for i in range(num_of_systems -2):
    new_config_number = system_SMs[i+2]  # Replace with the desired configuration number
    predicted_performance = model.predict(np.array([[new_config_number]]))
    linear_regression_prediction.append(float(predicted_performance[0]))




#End Linear Regression


#Logarithmic regression

logarithmic_prediction = []
logarithmic_prediction.append(float(sys.argv[1]))
logarithmic_prediction.append(float(sys.argv[2]))

# Sample data (replace with your data)
X = np.array([system_SMs[0], system_SMs[1]]).reshape(-1, 1)  # Configuration numbers
y = np.array([float(sys.argv[1]), float(sys.argv[2])])  # Performance numbers
log_transformer = FunctionTransformer(np.log, validate=True)
X_log = log_transformer.transform(X)

model = LinearRegression()
model.fit(X_log, y)


for i in range(num_of_systems -2):
    new_config_number = system_SMs[i+2] # Replace with the desired configuration number
    new_config_number_log = log_transformer.transform(np.array([[new_config_number]]))
    predicted_performance = model.predict(new_config_number_log)
    logarithmic_prediction.append(float(predicted_performance[0]))




#End Logarithmic



smallScale_realIPC = float(sys.argv[1])
#Calculate the correction factor
largeScale_expectedIPC = smallScale_realIPC*2;
#print("largeScale_expectedIPC: ", largeScale_expectedIPC)
largeScale_realIPC = float(sys.argv[2])
#print("largeScale_realIPC: ", largeScale_realIPC)
base_correction_factor = ((largeScale_realIPC - largeScale_expectedIPC)/largeScale_realIPC)
correction_factor = 1+((largeScale_realIPC - largeScale_expectedIPC)/largeScale_realIPC)
#Value for proportional scaling
proportional_scaling = []
#First element is the small scale model
proportional_scaling.append(smallScale_realIPC)
for i in range(num_of_systems-1):
    proportional_scaling.append(proportional_scaling[i]*2)


#scale-model value
scaleModel_IPC = []
scaleModel_IPC.append(smallScale_realIPC)
scaleModel_IPC.append(largeScale_realIPC)

postCliff_region=False
if (cliff_expected==0):
    #Because the first 2 are the scale models IPC and we want to predict the rest
    for i in range (len(system_SMs)-2):
        next_ipc = scaleModel_IPC[i+1]*2*correction_factor
        correction_factor = correction_factor + (correction_factor*base_correction_factor)
        scaleModel_IPC.append(next_ipc)
else:

    for i in range (len(system_SMs)-2):
        if system_SMs[i+2]==cliff_region_SM:
            #we are in the cliff region
            next_ipc = scaleModel_IPC[i+1] * 2 * correction_factor * (1 / (1 - (superLinear_factor / 100)))
            #Based correction factor
            correction_factor = 1+base_correction_factor
            postCliff_region=True
            scaleModel_IPC.append(next_ipc)
        else:
            
            next_ipc = scaleModel_IPC[i+1]*2*correction_factor
            correction_factor = correction_factor + (correction_factor*base_correction_factor)
            scaleModel_IPC.append(next_ipc) 

print("PowerLaw:", power_law_prediction)
print("Proportional:", proportional_scaling)
print("Logarithmic:", logarithmic_prediction)
print("Linear:", linear_regression_prediction)
print("Scale-Model:", scaleModel_IPC)

# Create the line graph with different colors
plt.plot(system_SMs, proportional_scaling, label='proportional', color='blue')
plt.plot(system_SMs, scaleModel_IPC, label='scaleModel', color='red', linestyle='dashed')
plt.plot(system_SMs, logarithmic_prediction, label='logarithmic', color='green')
plt.plot(system_SMs, power_law_prediction, label='powerLaw', color='black')
plt.plot(system_SMs, linear_regression_prediction, label='linear', color='purple')


# Add labels and a legend
plt.xlabel('#SM/#Chiplet')
plt.ylabel('IPC')
plt.legend()
plt.xticks(system_SMs, system_SMs)

# Adjust the layout to prevent overlap
plt.subplots_adjust(bottom=0.2)
# Display the graph
plt.savefig('scaleModel.png')
plt.show()
