import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
from scipy.signal import square
from scipy.interpolate import interp1d

class Skeletal_muscle:
    """ This class contains the parameters and equations describing dynamics of a skeletal muscle. The class is broken into following parts:
    1. Sarcolemma Model: - Contains ionic currents modeled with Hodgkin-Huxley-like conductance based equations. (UNDER CONSTRUCTION)
    2. Transverse tubule (T-tubule) Model: - Containts the T-tubule ion channels and voltage-gated calcium channels. (UNDER CONSTRUCTION)
    3. Sarcoplasmic Reticulum: - Consists of ryanodine receptors, IP3 receptors and SERCA Pumps. (UNDER CONSTRUCTION)
    4. Mitochondria (UNDER CONSTRUCTION)
    """
    
#     ########################################## Initialization (Under Construction) #################################
#     def __init__(i_step):
#         """This function is used to initialize model parameters and input patterns. For the first iteration of the code, thi"""
    ########################################## defining constants ##################################################
    
    ########### Electrical Parameters
    Cm = 1.                          # μFcm^-2; membrane capacitance
    
    ########### 1. Sodium conductance/channel
    g_max_Na = 804                   # mS.cm^-2; peak sodium conductance
    E_Na = 59.3                      # mV; Sodium channel Nernst Potential
    α_max_m = 0.288                  # (ms.mV)^-1; Maximum forward rate constant for m
    β_max_m = 1.38                   # ms^-1; Maximum backward rate constant for m
    α_max_h = 0.0081                 # ms^-1; Maximum forward rate constant for h
    β_max_h = 0.067                  # ms^-1; Maximum backward rate constant for h
    V_half_m = -46                   # mV; Half-maximum voltage for m
    K_alpha_m = 10                   # ms^-1; Steepness factor for m
    K_beta_m = 18                    # ms^-1: Slope factor for m
    V_half_h = -45                   # mV; Half-maximum voltage for h
    K_alpha_h = 14.7                 # (ms.mV)^-1; Steepness factor for h
    K_beta_h = 9                     # ms^-1; Slope factor for h
    V_half_S = -78                   # mV; Half-maximal voltage for S
    A_S = 5.8                        # mV; Steepness factorfor S
    V_tau = 90                       # Half-maximum voltage for τ_s
    
    ########## 2. Potassium conductance/channel
    g_max_K = 64.8                   # mS.cm^-2; peak potassium conductance
    E_K = -81.8                      # mV; Potassium channel Nernst Potential
    α_max_n = 0.0131                 # (ms.mV)^-1; Maximum forward rate constant for n
    β_max_n = 0.067                  # ms^-1; Maximum backward rate constant for n
    V_half_n = -40                   # mV; Half-maximum voltage for n
    K_alpha_n = 7                    # mV; steepness factor for n
    K_beta_n = 40                    # mV; Slope factor for n
    V_half_hk = -40                  # mV; Half-maximum voltage for hk
    A_hk = 7.5                       # mV; Steepness factor for hk
    
    ######### 3. Chloride conductance/channel
    g_max_Cl = 19.65                 # mS.cm^-2; Maximum conductance of Cl- channel
    E_Cl = -78.3                     # mV; Potassium channel Nernst Potential
    V_half_a = 70                    # mV; Half-maximum voltage for a
    A_a = 150                        # mV; Steepness factor for a
    
    
    
    
    ##################################### Functions ##############################################################
    def na_gating_vars(self, V):
        """This function takes the membrane voltage as input and returns α_m, β_m, α_h and β_h. Functions taken from Senneff and Lowrey 2020."""
    
        a_m = (self.α_max_m*(V - self.V_half_m))/(1 - np.exp((-(V - self.V_half_m))/(self.K_alpha_m)))
        b_m = self.β_max_m*np.exp(-(V - self.V_half_m)/(self.K_beta_m))
        a_h = self.α_max_h*np.exp(-(V - self.V_half_h)/(self.K_beta_h))
        b_h = (self.β_max_h)/(1 + np.exp(-(V - self.V_half_h)/(self.K_beta_h)))
        
        return a_m, b_m, a_h, b_h
    
    def na_slow_inactivation_vars(self, V):
        """This function takes membrane voltage as input and returns S_inf and τ_S variables of the sodium channel slow inactivation."""

        S_inf = 1/(1 + np.exp((V - self.V_half_S)/(self.A_S)))
        tau_S = (60)/(0.2 + 5.56*(np.power((V + self.V_tau), 2)/(100)))

        return S_inf, tau_S
    
    def k_gating_vars(self, V):
        """This function takes membrane voltage as input and returns α_n and β_n (Potassium gating variables). Functions taken from Senneff and Lowrey 2020."""
        a_n = (self.α_max_n*(V - self.V_half_n))/(1 - np.exp(-(V - self.V_half_n)/(self.K_alpha_n)))
        b_n = self.β_max_n*np.exp(-(V - self.V_half_n)/(self.K_beta_n))
        
        return a_n, b_n
    
    def h_K_inf(self, V):
        """This function takes membrane voltage V as input and returns h_K_inf parameter for slow K channel inactivation. Functions taken from Senneff and Lowrey 2020."""
        return 1/(1 + np.exp((V - self.V_half_hk)/(self.A_hk)))
    
    def tau_h_K(self, V):
        """This function takes membrane voltage V as input and returns the time constant τ_hK for K-channel slow inactivation"""
        return np.exp(-(V + 40)/25.75)
    
    def A_boltzmann(self, V):
        """This function takes membrane voltage V as input and returns the Boltzmann-function for activation of Cl- channel"""
        return 1/(1 + np.exp((V - self.V_half_a)/(self.A_a)))
    
    def I_Na(self, V, m, h, S):
        """Sodium current. Inputs: membrane potential (V) (in mV), activation variable (m), inactivation variable (h), slow-inactivation (S)
        Output: magnitude of sodium current (μA)"""
        
        return self.g_max_Na*np.power(m,3)*h*S*(V - self.E_Na)
    
    def I_Kdr(self, V, n, hk):
        """Potassium current. Inputs: membrane potential (V) (in mV), activation variable (n), slow inactivation (hk)
        Outputs: magnitude of potassium current (μA)"""
        
        return self.g_max_K*np.power(n,4)*hk*(V - self.E_K)
    
    def I_Cl(self, V):
        """Chloride current. Inputs: membrane potential (V) (in mV)
        Outputs: magnitude of chloride current (μA)"""
        
        A = self.A_boltzmann(V)
        
        return self.g_max_Cl*np.power(A,4)*(V - self.E_Cl)
    
    def I_pulse(self,t, i):
        """UNDER CONSTRUCTION: Current pulse between 50 - 100 ms"""
        return i if 50<t<100 else 0
    
    def I_eps(self, t):
        return self.y_fun(t)
        
    
    def dXdt(self, t, X):
        """Skeletal muscle excitation model containing Hodgkin-Huxley-style differential equations"""
        # unpacking the state-vector
        V, m, h, S, n, hk = X 
        
        # Voltage-dependent kinetic rates for sodium channel
        am, bm, ah, bh = self.na_gating_vars(V)
        
        # Voltage-dependent slow-inactivation parameters for sodium channel
        S_inf, tau_S = self.na_slow_inactivation_vars(V)
        
        # Voltage-dependent reaction kinetic rates for potassium channel
        an, bn = self.k_gating_vars(V)
        
        ## selecting the input-type
        if self.input_type == "pulse":
            I_in = self.I_pulse(t, self.i_inj)
        elif self.input_type == "eps":
            I_in = self.I_eps(t)
        
        # Hodgkin-Huxley equation
        dV_dt = (1/self.Cm)*(-self.I_Na(V, m, h, S) - self.I_Kdr(V, n, hk) - self.I_Cl(V) + I_in)
        
        # Sodium channel kinetics
        ## activation
        dm_dt = am*(1 - m) - bm*m
        
        ## inactivation
        dh_dt = ah*(1 - h) - bh*h
        
        ## slow-inactivation
        dS_dt = (S_inf - S)/tau_S
        
        # Potassium channel kinetics
        ## activation
        dn_dt = an*(1 - n) - bn*n
        
        ## slow-inactivation
        dhk_dt = (self.h_K_inf(V) - hk)/self.tau_h_K(V)
        
        return [dV_dt, dm_dt, dh_dt, dS_dt, dn_dt, dhk_dt]
    
    def sim_pulse(self, i_inj):
        """Simulates the muscle model for a pulse of amplitude i_inj"""
        # set the input type
        self.input_type = "pulse"
        
        # simulating the model in absence of inputs to get steady state
        ## set input to zero
        self.i_inj = 0.
        ## initial conditions
        X_init = [self.E_Cl+5., 0.001, 0.001, 0.001, 0.001, 0.001]
        ## initiating time vector for 1000 ms with Δt = 0.1 ms
        t_init = np.arange(0., 1000., 0.1) 
        ## solving the model in absence of inputs
        sol_init = solve_ivp(self.dXdt, [0, t_init[-1]], X_init, method="RK45")
        
        # simulating the model for a current pulse of amplitude i_inj and width = 50 ms
        ## set input to user-defined value
        self.i_inj = np.float64(i_inj)
        ## use the steady state values as initial conditions
        X_init = sol_init.y.T[-1,:]
        ## stop time
        t_final = 200 #ms
        ## solving the model for pulse input
        sol = solve_ivp(self.dXdt, [0, t_final], X_init, method="LSODA")
        
        
        return sol.t, sol.y.T, sol_init.t, sol_init.y.T
    
    def sim_eps(self, n_inputs, f_input, pulse_width, amplitude):
        """Simulates EPS protocol. Runs the model for square-wave input frequency f_input (Hz), #inputs = n_inputs, duration of pulse = pulse_width, amplitude = amplitude (μA)"""
        # set input type
        self.input_type = "pulse"
        # simulating the model in absence of inputs to get steady state
        ## set input to zero
        self.i_inj = 0.
        ## initial conditions
        X_init = [self.E_Cl+5., 0.001, 0.001, 0.001, 0.001, 0.001]
        ## initiating time vector for 1000 ms with Δt = 0.1 ms
        t_init = np.arange(0., 100., 0.1) 
        ## solving the model in absence of inputs
        sol_init = solve_ivp(self.dXdt, [0, t_init[-1]], X_init, method="RK45")
        
        # simulating the model for EPS protocol
        self.input_type = "eps"
        t_sim = np.arange(0., (n_inputs/f_input)*1e3, 0.0001)
        y = square(2*np.pi*f_input*t_sim*1e-3, duty=f_input*pulse_width*1e-3)
        y = amplitude*((y + 1)/2)
        self.y_fun = interp1d(t_sim, y)
        ## use the steady state values as initial conditions
        X_init = sol_init.y.T[-1,:]
        ## solving the model for pulse input
        sol = solve_ivp(self.dXdt, [0, t_sim[-1]], X_init, method="RK45")
        
        return sol.t, sol.y.T, sol_init.t, sol_init.y.T