import numpy as np
import random
import pandas as pd
from scipy.stats import lognorm
import matplotlib.pyplot as plt

# three life stages: egg->juvenile->adult
# takes 5 years to maturity and sexual maturity

# clusters are contained within the combined_cluster_stats.csv, which contains the cluster, average/max/min temperatures for taht cluster, and also average/max/min dissolved oxygen levels for that cluster 
# assume normnal distribution for both temperatures and dissolved oxygen

# calculates the number of eggs laid by a silver carp; utilizing a log-normal distribution 
def fecundity_value(target_mean, sigma):
    mu = np.log(target_mean) - (sigma**2 / 2)
    return np.random.lognormal(mean=mu, sigma=sigma, size=1) 

# load cluster data
cluster_data = pd.read_csv("combined_cluster_stats.csv")

class Population_Model:
    def __init__(self, cluster_id, starting_adult_pop):
        self.cluster_id = cluster_id
        self.eggs = 0
        self.juveniles_by_age = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self.adults = starting_adult_pop  # Starting number of mature adults that have magically appeared at the site
        #self.pop_count = self.juveniles + self.adults
        self.cluster_conditions = cluster_data[cluster_data['cluster'] == self.cluster_id].iloc[0]
        self.yearly_data ={'year': [], "juveniles":[], 'adults': []}
    
    def estimate_std(self, min_val, max_val):
        # assume distribution is approximately normal; with 95% of data between 2 stan deviations
        # if the cluster does not have a different min/max value, return -1
        if max_val != min_val:
            return (max_val - min_val)/4 
        else:
            return -1

    # def update_pop_count(self):
    #     self.pop_count += self.juveniles + self.adults
    
    # simulate a year and calculate the number of individuals in each life stage 
    def simulate_year(self, year):
        # for the current cluster, sample a (summertime) temperature from the temp distribution; fecundity occurs if >=15 degrees
        temp_std = self.estimate_std(self.cluster_conditions['MIN_SUMMER_TEMP'], self.cluster_conditions['MAX_SUMMER_TEMP'])
        if temp_std == -1:
            sample_temp = self.cluster_conditions["AVG_SUMMER_TEMP"]
        else:
            sample_temp = np.random.normal(self.cluster_conditions['AVG_SUMMER_TEMP'], temp_std)
        
        # sample a dissolved oxygen value; any value >=7 mg/L is beneficial thus no impact on survival; if <7, each percentage below this value will increase mortality rate; if drawn dissolved oxygen is 6, then there is a 10% chance of mortality
        oxygen_std = self.estimate_std(self.cluster_conditions['MIN_OXYGEN'], self.cluster_conditions['MAX_OXYGEN'])
        if oxygen_std == -1:
            sample_oxygen = self.cluster_conditions['AVG_OXYGEN']
        else:
            sample_oxygen = np.random.normal(self.cluster_conditions['AVG_OXYGEN'], oxygen_std)

        # eggs are produced if the summer sample_temp is >= 15 degrees
        if sample_temp >= 15:
            # assume half population of adults are female
            for adult in range(int(self.adults / 2)):
                self.eggs += fecundity_value(269388, 0.5)[0]

            # calculate survival of eggs; from literature, 80% are fertilized successfully and hatch, then around 1 percent survive to become fry/juveniles 
            self.juveniles_by_age[1] += np.round(self.eggs * 0.80 * 0.01)
            # reset number of eggs
            self.eggs = 0
            # calculate survival of juveniles and adults 
            self.calculate_survival(sample_oxygen) 
            # increment ages of juveniles that have survived
            # if juveniles age == 5, they become adults 
            new_adults = self.juveniles_by_age[5]
            self.adults += new_adults
            # age the rest of the juveniles 
            for age in reversed(range(1,6)):
                # reverse the order so that we wont incorrectly increment 
                if age == 5:
                    self.juveniles_by_age[age] = 0
                else:
                    self.juveniles_by_age[age+1] = self.juveniles_by_age[age]
            self.juveniles_by_age[1] = 0
        else:
            # no spawning occurs, but still increment juveniles ages and calculate survivability
            self.calculate_survival(sample_oxygen)
            # increment ages of juveniles that have survived
            # if juveniles age >=5, become adults 
            new_adults = self.juveniles_by_age[5]
            self.adults += new_adults
            for age in reversed(range(1,6)):
                if age == 5:
                    self.juveniles_by_age[age] = 0
                else:
                    self.juveniles_by_age[age+1] = self.juveniles_by_age[age]
            self.juveniles_by_age[1] = 0

        self.yearly_data['year'].append(year)
        self.yearly_data['juveniles'].append(sum(self.juveniles_by_age.values()))
        self.yearly_data['adults'].append(self.adults)
    
    # Calculate the survival rate of the current model 
    def calculate_survival(self, oxygen_level):
        # calculate survival of juveniles; split the 85% winter surviability between adults and juveniles, but juveniles have
        # the minimal survival rate; also if dissolved oxygen level is <7, then mortality increases
        total_survival_rate = 0.85
        adult_survival_portion = random.uniform(0.5, 0.7)
        adult_survival_rate = total_survival_rate * adult_survival_portion
        juvenile_survival_rate = total_survival_rate * (1 - adult_survival_portion)

        if oxygen_level < 7:
            mortality_increase = (7-oxygen_level)*0.1
        else:
            mortality_increase = 0

        adjusted_adult_survival = max(0, adult_survival_rate - mortality_increase)
        adjusted_juvenile_survival = max(0, juvenile_survival_rate - mortality_increase)

        # Apply adjusted survival rates
        for age in range(1, 6):
            if age == 5:
                # Transition age 5 juveniles to adults before applying survival rates
                new_adults = self.juveniles_by_age[age]
                self.adults += new_adults
                self.juveniles_by_age[age] = 0
            else:
                # Apply juvenile survival rate and age juveniles
                self.juveniles_by_age[age + 1] = np.round(self.juveniles_by_age[age] * adjusted_juvenile_survival)

        # Apply adult survival rate
        self.adults = np.round(self.adults * adjusted_adult_survival)

        # Prepare for new juveniles from this year's eggs, resetting the youngest juveniles
        self.juveniles_by_age[1] = 0

    def total_population(self):
        return sum(self.juveniles_by_age.values()) + self.adults

# # clusters are its own columnn in the dataset
# cluster_data['20_year_pop'] = np.nan
# for cluster_id in cluster_data['cluster']:
#     model = Population_Model(cluster_id, starting_adult_pop=100)  
#     # simulate 20 years; append the 
#     for _ in range(20):
#         model.simulate_year()
#      # Append the final population to the 'cluster_data'
#     final_pop = model.total_population()
#     cluster_data.loc[cluster_data['cluster'] == cluster_id, '20_year_pop'] = final_pop

# # Save the updated DataFrame
# cluster_data.to_csv('cluster_data_with_20_year_simulation.csv', index=False)
    


# clusters are its own columnn in the dataset
model = Population_Model(cluster_id=3, starting_adult_pop=100)
for year in range(1,21):
    model.simulate_year(year)
# Data from the simulation
years = model.yearly_data['year']
juveniles = model.yearly_data['juveniles']
adults = model.yearly_data['adults']

plt.figure(figsize=(10, 6))
plt.plot(years, juveniles, label='Juveniles', marker='o', linestyle='-', color='blue')
plt.plot(years, adults, label='Adults', marker='o', linestyle='-', color='green')

plt.title(f'Population Dynamics for Cluster 3 Over Time')
plt.xlabel('Year')
plt.ylabel('Population')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# cluster_data['20_year_pop'] = np.nan
# for cluster_id in cluster_data['cluster']:
      
#     # simulate 20 years; append the 
#     for _ in range(20):
#         model.simulate_year()
#      # Append the final population to the 'cluster_data'
#     final_pop = model.total_population()
#     cluster_data.loc[cluster_data['cluster'] == cluster_id, '20_year_pop'] = final_pop

# # Save the updated DataFrame
# cluster_data.to_csv('cluster_data_with_20_year_simulation.csv', index=False)