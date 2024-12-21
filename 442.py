import math
import random
import numpy as np
import matplotlib.pyplot as plt

coordinates = []
with open(r"C:\Users\tombe\Documents\Master\Stochastic Simulation\Assignment 3\pcb442filter.txt", "r") as file:
    for line in file:
        line = line.strip()
        
        parts = line.split()
        if len(parts) == 3:
            _, x, y = parts
            x_coord = float(x)
            y_coord = float(y)
            coordinates.append((x_coord, y_coord))

#eli51 = r'C:\Users\tombe\Documents\Master\Stochastic Simulation\Assignment 3\eil51.tsp.txt'
opt_route = []
with open(r"C:\Users\tombe\Documents\Master\Stochastic Simulation\Assignment 3\pcb442.opt.tour.txt", "r") as file:
    for line in file:
        opt_route.append(int(line)-1)  


#distances between cities
def distance(coord1, coord2):
    return math.sqrt((coord2[0]-coord1[0])**2 + (coord2[1]-coord1[1])**2)

#now save all distances between all cities in a n by n matrix
all_distances = np.zeros([len(coordinates), len(coordinates)], dtype=float) 
for i in range(len(coordinates)):
    for j in range(len(coordinates)):
        all_distances[i][j] = distance(coordinates[i], coordinates[j])

#length of the full route
def length_of_route(route):
    total_distance = 0
    #distances are already calculated. This piece of code finds the matrix entries and adds them corresponding to the route
    for i in range(len(route)-1):
        total_distance += all_distances[route[i]][route[i+1]]
    total_distance += all_distances[route[-1]][route[0]] #this makes sure you return to the starting city
    return total_distance

#define functions that change the route in different ways:
def inverting(current_route):
    i, j = sorted(random.sample(range(len(current_route)), 2))
    new_route = np.concatenate([current_route[:i], current_route[i:j+1][::-1], current_route[j+1:]])
    new_route = new_route.astype(int) 
    return new_route

def two_opt(current_route):
    i, j = sorted(random.sample(range(len(current_route)), 2))
    new_route = current_route[:i+1] + current_route[i+1:j+1][::-1] + current_route[j+1:]
    return new_route

def swapping(current_route):
    i, j = random.sample(range(len(current_route)), 2)
    new_route = current_route.copy()
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

# #function for the simulated annealing process. It will then check if the new route is better than the last one. If yes, accept straight away 
# # if not, it will check if the probability corresponding to difference between the old and new route is higher than some random number (the acceptance process)
def simulated_annealing(initial_route, n_iter, cooling_schedule, initial_temperature, route_edit):
    current_route = initial_route.copy()
    current_length = length_of_route(initial_route)
    best_route = initial_route.copy()
    best_length = length_of_route(best_route)

    for iter in range(n_iter):
        temperature = cooling_schedule(initial_temperature, iter, n_iter)
        new_route = route_edit(current_route)
        new_route_length = length_of_route(new_route)

        if new_route_length < current_length:
            current_route = new_route
            current_length = new_route_length
            if current_length < best_length:
                best_length = current_length
                best_route = current_route

        else:
            #with this next part, you just sometimes choose to accept a worse route 
            #we came from the last if-statement where we accepted the new route if it was better than the old one.
            # Here we know already it's a worse solution and the difference tells you how much worse. 
            difference = new_route_length - current_length                                                            
            probability = math.exp(-difference/temperature) #so the 'energy' from the boltzmann distribution here is the difference between old and new route
            if np.random.rand() < probability: #now randomly allow this worse solution to be further investigated (or not)
                current_route = new_route
                current_length = new_route_length
                if current_length < best_length:
                    best_length = current_length
                    best_route = current_route
            
    return best_route, best_length, temperature

#define cooling schedules:
def linear_cooling(initial_temperature, iteration, n_iter):
    return max(0, initial_temperature * (1 - iteration / n_iter))
    #return initial_temperature - alpha * iteration

def geometric_cooling(initial_temperature, iteration, n_iter):
    ratio = (initial_temperature / 1e-9) ** (1 / n_iter)  
    return max(0, initial_temperature * (ratio ** (-0.2*iteration)))

def logarithmic_cooling(initial_temperature, iteration, n_iter):
    return  max(0, initial_temperature * (1 - np.log(iteration + 1) / np.log(n_iter + 1)))

def exponential_decay(initial_temperature, iteration, n_iter):
    return initial_temperature / (1+np.exp((iteration - n_iter/2.5) / (3*n_iter/100))**0.5)

#Random initial route 
city_indices = np.arange(0,len(coordinates))
initial_route = random.sample(range(len(city_indices)), len(city_indices))

#initial_route = np.arange(0,len(coordinates)) #just from 0 to the last one

n_iter = 10**3 #so watch out with changing this one
initial_temperature = 1000


best_route, best_length, temp = simulated_annealing(initial_route=initial_route, n_iter=n_iter, initial_temperature=initial_temperature, cooling_schedule=exponential_decay, route_edit= two_opt)

print(temp)
print(f"Best route found: {best_route}")
print(f"Distance of best route: {best_length}")

#Plot the cooling schedules
iterations = np.linspace(0, 1 * n_iter, initial_temperature)
temp2 = [exponential_decay(initial_temperature, iter, n_iter) for iter in iterations]
temp3 = [linear_cooling(initial_temperature, iter, n_iter) for iter in iterations]
temp4 = [logarithmic_cooling(initial_temperature, iter, n_iter) for iter in iterations]
temp5 = [geometric_cooling(initial_temperature, iter, n_iter) for iter in iterations]
plt.plot(iterations, temp2, label='Exponential Decay', color='blue')
plt.plot(iterations, temp3, label='Linear' )
plt.plot(iterations, temp4, label='Logarithmic' )
plt.plot(iterations, temp5, label='Geometric' )
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Temperature', fontsize=14)
plt.grid()
plt.legend(fontsize=12)
plt.show()


#plot route
x_opt = [coordinates[city][0] for city in opt_route]
y_opt = [coordinates[city][1] for city in opt_route]
x_coords = [coordinates[city][0] for city in best_route]
y_coords = [coordinates[city][1] for city in best_route]
#end at start:
x_coords.append(coordinates[best_route[0]][0])
y_coords.append(coordinates[best_route[0]][1])

plt.plot(x_coords, y_coords, marker='o')
plt.plot(x_opt, y_opt, marker='o', color='red', alpha=0.4)
plt.xlabel('x-coordinate of city')
plt.ylabel('y-coordinate of city')
plt.show()

#MULTIPLE RUNS
import scipy.stats as st  

schedules = [linear_cooling, geometric_cooling, logarithmic_cooling, exponential_decay]
edits = [swapping, inverting, two_opt]

n_iter_values = [10**3, 10**5, 10**7]  # Different Markov chain lengths to test
initial_temperature = 1000
runs_per_setting = 5  # number of times to run each combination

for n_iter_test in n_iter_values:
    for schedule in schedules:
        for route_edit_func in edits:
            best_route_lengths_all_runs = []
            best_routes_all_runs = []

            for run_index in range(runs_per_setting):
                initial_route = random.sample(range(len(city_indices)), len(city_indices))
                best_route, best_length, temp = simulated_annealing(
                    initial_route=initial_route,
                    n_iter=n_iter_test,
                    initial_temperature=initial_temperature,
                    cooling_schedule=schedule,
                    route_edit=route_edit_func
                )
                best_routes_all_runs.append(best_route)
                best_route_lengths_all_runs.append(best_length)

            # Compute statistics
            lengths_array = np.array(best_route_lengths_all_runs)
            mean_length = np.mean(lengths_array)
            std_length = np.std(lengths_array, ddof=1)  # sample standard deviation
            
            # 95% confidence interval
            conf_level = 0.95
            alpha = 1 - conf_level
            df = runs_per_setting - 1
            t_crit = st.t.ppf(1 - alpha/2, df) if runs_per_setting > 1 else np.nan
            margin_of_error = t_crit * (std_length / np.sqrt(runs_per_setting)) if runs_per_setting > 1 else np.nan
            ci_lower = mean_length - margin_of_error if runs_per_setting > 1 else mean_length
            ci_upper = mean_length + margin_of_error if runs_per_setting > 1 else mean_length

            # Find the best route among the runs
            min_best_route_length = np.min(best_route_lengths_all_runs)
            min_best_route_index = best_route_lengths_all_runs.index(min_best_route_length)
            min_best_route = best_routes_all_runs[min_best_route_index]

            # Print results
            print(f"Markov Chain Length: {n_iter_test}")
            print(f"Schedule: {schedule.__name__}, Route Edit: {route_edit_func.__name__}")
            print(f"All run lengths: {best_route_lengths_all_runs}")
            print(f"Mean: {mean_length:.4f}, Std: {std_length:.4f}")
            if runs_per_setting > 1:
                print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"Best route found: {min_best_route}")
            print(f"Distance of best route: {min_best_route_length}")
            print(f"Final temperature: {temp}\n")

            # Plot the best route
            x_coords = [coordinates[city][0] for city in min_best_route]
            y_coords = [coordinates[city][1] for city in min_best_route]
            # End at start
            x_coords.append(coordinates[min_best_route[0]][0])
            y_coords.append(coordinates[min_best_route[0]][1])

            plt.figure(figsize=(8,6))
            plt.plot(x_coords, y_coords, marker='o', label='Best Found')
            plt.plot(x_opt, y_opt, marker='o', color='red', alpha=0.4, label='Optimal (reference)')
            plt.xlabel('x-coordinate of city', fontsize=14)
            plt.ylabel('y-coordinate of city', fontsize=14)
            plt.title(f'Best route with {schedule.__name__}, {route_edit_func.__name__}, n_iter={n_iter_test}', fontsize=17)
            plt.legend(fontsize=12)
            plt.grid()
            plt.show()
