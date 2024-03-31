import numpy as np
import random
from scipy.stats import lognorm
# three life stages: egg->juvenile->adult
# takes 5 years to maturity and sexual maturity


# calculates the number of eggs laid by a silver carp; utilizing a log-normal distribution 
def fecundity_value(SilverCarp, target_mean, sigma):
    mu = np.log(target_mean) - (sigma**2 / 2)
    return np.random.lognormal(mean=mu, sigma=sigma, size=1) 

# class carp_life_stage:
#     def __init__(self, age):
#         self.age = age
    
#     def increment_age(self):
#         self.age +=1
    
#     class juvenile_carp():

    
#     class adult_carp():

class Population_Model:
    def __init__(self, egg_survival, juvenile_survival):
        self.egg_survival = egg_survival
        self.juvenile_survival = juvenile_survival
        self.eggs = 0
        self.juveniles = 0
        self.adults = 10  # Starting number of mature adults that have magically appeared at the site
        self.pop_count = self.juveniles + self.adults
    
    def update_pop_count(self):
        self.pop_count += self.juveniles + self.adults

    # Calculate the survival rate of the current model 
    def calculate_survival(self, conditions):
        # calculate survival of eggs; from literature, 80% are fertilized successfully and hatch, then around 1 percent survive to become fry 
        self.eggs = np.round(self.eggs * 0.80 * 0.01)

        # calculate survival of juveniles; split the 85% winter surviability between adults and juveniles, but juveniles have
        # the minimal survival rate 
        juvenile_survival = random.uniform(0.1, 0.4)
        self.juveniles = np.round(self.juveniles * juvenile_survival)

        # calculate adult survival
        adult_survival = 0.85 - juvenile_survival
        self.adults = np.round(self.adults * adult_survival)

    # simulate a year and calculate the number of individuals in each life stage 
    def simulate_year(self, favorable_conditions):
        self.calculate_survival()
        # increment ages
        eggs_produced = np.round(self.adults/2) * fecundity_value(269388, 0.5)
    
        # increment juvenile age

        # Juveniles becoming adults
        new_adults = np.random.binomial(juveniles, juvenile_survival_rate)
        self.adults += new_adults
    
    # Run the simulation at the inputted coordinate for the given years 
    def run_simulation(self, years):
        for year in range(years):
            # Assume 50% chance each year is favorable
            self.simulate_year(favorable_conditions=np.random.rand() > 0.5)
            print(f"Year {year+1}: Adult Population = {self.adults}")

# Example usage
model = Population_Model(egg_survival=0.05, juvenile_survival=0.10, fecundity=156312)
model.run_simulation(20)