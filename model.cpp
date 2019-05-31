// This model is based on Soto-Trevino et all 2005 and adds Q10's.
// This version includes variable time steps. There are two dt's:
// ---- dt is the current time step, which will change as needed. 
// ---- dt_hist is the time step for recording history in the sim.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "model.h"
#include <vector>
#include <cassert>
#include <iostream>
#include <numeric>
#include <sys/file.h>
#include <unistd.h>
#include <random>
#include <algorithm>

using namespace std;

// Model parameters //
const bool   use_Liu_CaS         = false;           // Substitute Liu 1998's inactivating CaS channel instead of Soto-Trevino's non-inactivating CaS channel.
const double epoch_length        = 1;              // [seconds] An "epoch" is a slice of simulation time. After each epoch is run, a check for whether the system has begun to repeat its state will be done.
const double min_sim_time        = 0;              // Set a floor for simulation time. 

const short  num_compartments    = 2;              // Number of compartments per neuron. Compartment 1 is soma. Compartment 2 is spike initiation zone.

const bool temperature_dependent_reversal_potentials = true;   // Calculate reversal potential of Ca++, Na+ and K+ (as well as mixed cation channels) depending on temperature.
      bool constant_Ca_reversal_potential            = false;  // Used only for testing 
      bool constant_Ca_concentration                 = false;  // Used only for testing. Internal [Ca++] held constant at Ca_steady_state level, despite CaT and CaS currents as if instantly buffered.

vector <double> g_axial    (num_neurons, 0); // [num_neurons]; // Axial conductance 
vector <double> g_axial_adj(num_neurons, 0); // ... After adjustment for Q10
double g_gap;                    // Gap junction conductance
double g_gap_adj;                // ... After adjustment for Q10
double capacitance[num_neurons][num_compartments];

const double R = 8.31447215;      // [J/mol/K] Ideal gas constant
const double reference_Temp = 273.15 + 11;//10.919;       // [Kelvin] Temperature (Soto-Trevino's animals are tested at 18C. Buchholtz et al 1992 use 10C.) 
      double T;                   // [Kelvin]  Temperature 
const double z = +2.0;            // Charge of Ca++.
const double F = 96485.3399;      // [C/mol] Faraday's constant
const double Ca_out             = 13000.0*pow(10.0,-6); // [M]
const double Ca_steady_state    = 0.5*pow(10.0,-6);     // [M]
double tau_Ca[num_neurons];
double F_Ca[num_neurons]; 

// Reversal potentials - not constant since they change with temperature.
double E_Na; // [V] 
double E_K ;
double E_H ;
double E_proc;
double E_Ca; // (Placeholder. Nernst calculation later.)

double E_all[2][num_channels];  //[V] Store reversal potentials in vector which matches the channel vector.

vector<vector<double> >  g_max_all_adj           (num_neurons, vector<double>(num_channels, 0)); // [S]  Max conductance values, adjusted for Q10's.
vector<vector<double> >  g_max_all               (num_neurons, vector<double>(num_channels, 0)); // [S]  g_max values at reference temperature.

vector<vector<double> >  g_leaks    (2, vector<double>(2, 0));    // Leak conductances    (Before adjustment for temperature. After conversion to whole units.)
vector<vector<double> >  g_leaks_adj(2, vector<double>(2, 0));    // Leak conductances    (After  adjustment for temperature and conversion to whole units.)
                                    
// Channel                                     Na   K(AIS)   CaT   CaS   nap   h    K(soma)  KCa   A    proc
const double a_exponents[2][num_channels] = {{ 3,   4,       3,    3,    3,    1,   4,       4,    4,   1    },    // AB  NOTE: Swapped values for m_A in this version.
                                             { 3,   4,       3,    3,    3,    1,   4,       4,    3,   1    }};   // PD
const double b_exponents[num_channels] =     { 1,   0,       1,    0,    1,    0,   0,       0,    1,   0    };    // Both AB and PD
                                    

double Q_alphas[2][num_channels]; // Rate multiplier for channel opening rate. (This is the Q10 with temperature already taken into consideration.)
double Q_betas [2][num_channels];
double Q_g_bars[num_channels];

double delta_Temperature;

double Q_Ca;
double Q_gap;
double Q_leak;
double Q_axial[num_neurons];

// Leak reversal potentials
//              Soma  AIS
const double E_leak[2][2] = {{ -0.050,   -0.060},   // AB [V]
                             { -0.055,   -0.055}};  // PD
                             
double I_CaT[num_neurons], I_CaS[num_neurons];
double g_channels[num_neurons][num_channels]; // channel conductance at present time.    
                             
const int test_steps = 4; // Number of provisional steps being tested. (Three for half step method - the two half steps and the full step.)
// State variables. Storing current state and state from experimental steps here.
double voltages[num_neurons][num_compartments][test_steps];
double Ca[num_neurons][test_steps]; // Ca++ concentration.
double activations  [num_neurons][num_channels][test_steps];
double inactivations[num_neurons][num_channels][test_steps];
double currents     [num_neurons][num_channels][test_steps];

double sim_time; // Total elapsed simulation time at current step. (Does not track experimental steps.)


enum integrationMethods {forwardEuler, expEuler, rk4, rk4_withQ10};

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//          model()             %
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
network_stats model(double temperature, double Q10_alphas[2][num_channels], double Q10_betas[2][num_channels], double Q10_g_bars[num_channels], double Q10_Ca, double Q10_axial[num_neurons], double Q10_gap, double Q10_leak,
                    double sim_length, double dt, double dt_hist, bool cluster, bool randomize_initial_conductances, int g_max_set, bool constant_Ca, int rng_seed){

    #ifdef testing
        cout << "Testing mode on." << endl;
    #else
        if (!cluster) {cout << "Testing mode off." << endl;}
    #endif
    
    if (constant_Ca) { // Used for testing model. These values are normally false.
        constant_Ca_reversal_potential = true;
        constant_Ca_concentration      = true;
    }
    
    // Print warnings for certain settings
    if (constant_Ca_reversal_potential) {cout << "WARNING: Using constant_Ca_reversal_potential."              << endl;}
    if (constant_Ca_concentration     ) {cout << "WARNING: Using constant_Ca_concentration (internal [Ca++])." << endl;}
    
    int   time_step_hist = 0; // Number of intervals of size dt_hist which have passed between start of sim and last recording of state history. ( if ((time_step_hist+1)*dt_hist < sim_time) then {record some history} ) 
    sim_time = 0;
    bool   at_ss = false; // Are we at steady state yet?
    //double time_Min_ss = 3*1000; //[ms] Minimum simulation time allowed before declaration of steady state is allowed.
    //int index_initial_ss = -1;
    //int index_final_ss   = -1;
    
    // Reversal potentials at reference temperature. 
    E_Na   =  50.0*pow(10.0,-3); // [V] 
    E_K    = -80.0*pow(10.0,-3);
    E_H    = -20.0*pow(10.0,-3);
    E_proc =   0.0*pow(10.0,-3);
    E_Ca   =  9999999999999; // (Placeholder. Nernst calculation later.)

    // Maximal conductances (Before adjustment for temperature and coversion to whole units.)
    //                            Channels are:
    //         (AIS) NA    K      (soma) CaT    CaS  nap    H       K        KCa      A       proc
    //g_max_all = {{ 300,  52.5,         55.2,  9,   2.7,   0.054,  1890,    6000,    200,    570 },   // AB  [mS] Conductance values for all (non-leak) channels.
    //              {1100,  150,         22.5,  60,  4.38,  0.219,  1576.8,  251.85,  39.42,  0   }};  // PD 
    if (g_max_set){
        FILE *g_max_file;
        if (cluster){
            g_max_file = fopen("/home/caplan/Q10_project/random_model/g_maxes.txt","r");
        } else {
            g_max_file = fopen("/home/jon/Documents/Q10_project/model/g_maxes.txt","r");
        }
        if (g_max_file == NULL){cout << "Could not open g_maxes.txt file. Exiting." << endl; exit(-1);}
        for (int set = 1; set<= g_max_set; set++){
            for (int neuron = 0; neuron < 2; ++neuron){
                for(int conductance = 0; conductance<10; ++conductance){
                    double the_value = 0;
                    int result = fscanf(g_max_file, "%lf ", &the_value );
                    if (result !=1) {cout << "Could not read item in g_maxes.txt file. Exiting." << endl; exit(-1);}
                    if (set == g_max_set) {
                        //the_value *= 1000; // *1000 converts from mS to uS.
                        g_max_all[neuron][conductance] = the_value;
                       //cout << "wrote value: " << the_value << endl;
                    } // Save the values if we're in the right line of the file.
                }
            }
        }
        
    }else{
        g_max_all[AB][0] = 300;  // Alternate version because compiler on cluster complains about above version.
        g_max_all[AB][1] = 52.5;
        g_max_all[AB][2] = 55.2;
        g_max_all[AB][3] = 9;
        g_max_all[AB][4] = 2.7;
        g_max_all[AB][5] = 0.054;
        g_max_all[AB][6] = 1890;
        g_max_all[AB][7] = 6000;
        g_max_all[AB][8] = 200;
        g_max_all[AB][9] = 570;
        
        g_max_all[PD][0] = 1100;
        g_max_all[PD][1] = 150;
        g_max_all[PD][2] = 22.5;
        g_max_all[PD][3] = 60;
        g_max_all[PD][4] = 4.38;
        g_max_all[PD][5] = 0.219;
        g_max_all[PD][6] = 1576.8;
        g_max_all[PD][7] = 251.85;
        g_max_all[PD][8] = 39.42;
        g_max_all[PD][9] = 0;
    }

    if (randomize_initial_conductances){

	std::mt19937 generator (rng_seed);  // mt19937 is a standard mersenne_twister_engine

	cout << "g_maxes" << endl;

        for (int neuron=0; neuron<num_neurons; ++neuron){
            for (int channel = 0; channel < num_channels; ++channel){
                double rand = (float)generator()/(float)generator.max();
                g_max_all[neuron][channel] *=  rand*2.0;  // Randomize around a known burster.
		cout << g_max_all[neuron][channel] <<  "  ";
            }
	    cout << "     ";
        }
	cout << endl;
    }

    // [mS] Leak conductances [AB-soma, AB-AIS; PD-soma PD-AIS] (Note: Compartment order different than for g_max_all.)
    g_leaks[AB][0] = 0.045;
    g_leaks[AB][1] = 0.0018;
    g_leaks[PD][0] = 0.105;
    g_leaks[PD][1] = 0.00081;
    

    T = temperature+273.15; // [Kelvin]  

    g_axial[AB]    = 0.3 *pow(10.0,-6); // [S] Conductance between the two compartments of the AB neuron.
    g_axial[PD]    = 1.05*pow(10.0,-6); // [S] Conductance between the two compartments of the PD neuron.
    g_gap          = 0.75*pow(10.0,-6); // [S] Conductance between the soma of the two neurons.
    
    
    capacitance[AB][AIS ] =  1.5*pow(10.0,-9); // [F] Capacitance of each compartment.
    capacitance[AB][Soma] =  9.0*pow(10.0,-9); // [F]
    capacitance[PD][AIS ] =  6.0*pow(10.0,-9); // [F]
    capacitance[PD][Soma] = 12.0*pow(10.0,-9); // [F]
    
    // Constants for Ca++ dynamics
    tau_Ca[AB]         = 0.303;       // [s]
    tau_Ca[PD]         = 0.300;       // [s]   // 
    F_Ca[AB]   = 0.4177777*pow(10.0,3);  // [M/A]
    F_Ca[PD]   = 0.5150685*pow(10.0,3);  // [M/A]

    for (int neuron = 0; neuron < num_neurons; ++neuron){
        for (int channel = 0; channel<num_channels; ++channel){
            g_max_all[neuron][channel] *= pow(10.0,-6);  // [S] convert values to whole units.
            g_max_all_adj[neuron][channel] =  g_max_all[neuron][channel]; 
        }
    }
    for (int neuron = 0; neuron < num_neurons; ++neuron){
        for (int compartment = 0; compartment<num_compartments; ++compartment){
            g_leaks[neuron][compartment] *= pow(10.0,-6);  // [S] convert values to whole units.
        }
    }    
                               
    // Find the Q values from the temperature and the Q10s.
    //         Q10 = (R1/R2)^(10/deltaT)
    //         Q   = R1/R2
    //         Q10 = Q^(10/deltaT)
    //         Q   = Q10^(deltaT/10)
    delta_Temperature = T - reference_Temp; 
    for(int channel = 0; channel<num_channels; ++channel){
        for (int neuron = 0; neuron<num_neurons; ++neuron){
            Q_alphas[neuron][channel] = pow(Q10_alphas[neuron][channel], (delta_Temperature/10.0));
            Q_betas [neuron][channel] = pow(Q10_betas [neuron][channel], (delta_Temperature/10.0));
        }
        Q_g_bars[channel] = pow(Q10_g_bars[channel], (delta_Temperature/10.0));
    }
    Q_Ca        = pow(Q10_Ca,        (delta_Temperature/10.0)); // Represents the buffering of Ca++, not the rate of entry of Ca++ into the cell.
    Q_gap       = pow(Q10_gap,       (delta_Temperature/10.0));
    Q_leak      = pow(Q10_leak,      (delta_Temperature/10.0));
    Q_axial[AB] = pow(Q10_axial[AB], (delta_Temperature/10.0));
    Q_axial[PD] = pow(Q10_axial[PD], (delta_Temperature/10.0));
    
    // Adjust model paramaters by Q factors.
    g_axial_adj[AB] = g_axial[AB] * Q_axial[AB];
    g_axial_adj[PD] = g_axial[PD] * Q_axial[PD];
    g_gap_adj      = g_gap        * Q_gap;

    for (int neuron = 0; neuron<num_neurons; ++neuron){
        for(int compartment=0; compartment<num_compartments; ++compartment){
            g_leaks_adj[neuron][compartment] = g_leaks[neuron][compartment] * Q_leak;
        }
    }
    
    for (int neuron = 0; neuron<num_neurons; ++neuron){
        for(int channel = 0; channel<num_channels; ++channel){
            g_max_all_adj[neuron][channel]    = g_max_all[neuron][channel] *  Q_g_bars[channel];
        }
    }
    
    //%%%%%%%%%%%%%%%%%%%%%%
    // Starting conditions %
    //%%%%%%%%%%%%%%%%%%%%%%
    
    bool spiking [num_neurons] = {false, false}; // Spiking  state of [AB PD] 
    bool bursting[num_neurons] = {false, false}; // Bursting state of [AB PD]
    
    
    // Initial values from Farzan.
                            // AB            Na             K(AIS)               CaT         CaS          nap           h             K(soma)        KCa           A        proc
    double initial_AB_activations  [10] = { 0.008069,     0.045224,            0.029220,   0.035182,    0.054560,    0.037455,    0.045529,    0.016805,   0.065484,   0.000004};
    double initial_PD_activations  [10] = { 0.007944,     0.044920,            0.028679,   0.034632,    0.053695,    0.037369,    0.045019,    0.025391,   0.064513,   0.000004};
    double initial_AB_inactivations[10] = { 0.560552,     1.000000,            0.886411,   1.000000,    0.548122,    1.000000,    1.000000,    1.000000,   0.204429,   1.000000};
    double initial_PD_inactivations[10] = { 0.564511,     1.000000,            0.884171,   1.000000,    0.544207,    1.000000,    1.000000,    1.000000,   0.209055,   1.000000};
    for (int channel = 0; channel<num_channels; ++channel) {
        activations  [AB][channel][0] = initial_AB_activations  [channel];
        inactivations[AB][channel][0] = initial_AB_inactivations[channel];
        activations  [PD][channel][0] = initial_PD_activations  [channel];
        inactivations[PD][channel][0] = initial_PD_inactivations[channel];
    }
    voltages[AB][Soma][0] = -0.050069767; // [V]
    voltages[PD][Soma][0] = -0.050209066; // [V]
    voltages[AB][AIS] [0] = -0.050152648; // [V]
    voltages[PD][AIS] [0] = -0.050236002; // [V]

    Ca[AB][0] = 0.905226*pow(10.0,-6);   // [M]
    Ca[PD][0] = 1.405941*pow(10.0,-6);   // [M]
    
    int num_state_variables_per_neuron = num_compartments + num_channels*2 + 1; // These state variables are for Voltage, (in)activations, and [Ca++], respectively.
    assert (num_state_variables_per_neuron == 23);
    //const double state_ref[23]    = { 0.025, 0.025, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0000005 };  // This is a scaling factor to normalize the Values repesent: Voltage [V], (in)activation, [Ca++] [M],,
    
    // History variables
    static const int save_slots = (int) ceil(sim_length/dt_hist) +1 ;
    if (!cluster) {cout << "num save slots = " << save_slots << endl;}

    vector<vector<vector<double> > > V_hist (num_neurons, vector < vector<double> > (num_compartments, vector<double>(save_slots,0)));
    // Remove (most) history recording code when running on cluster. 
    #ifdef desktop
        vector<vector<vector<double> > > m_hist (num_neurons, vector < vector<double> > (num_channels,     vector<double>(save_slots,0)));
        vector<vector<vector<double> > > h_hist (num_neurons, vector < vector<double> > (num_channels,     vector<double>(save_slots,0)));
        vector<vector<double> >          Ca_hist(num_neurons,                                              vector<double>(save_slots,0));
        vector<vector<vector<double> > > state_hist     (num_neurons, vector < vector<double> > (num_state_variables_per_neuron, vector<double>(save_slots,0)));   // Holds all state history in one arrray.
        //vector<vector<vector<double> > > dist_state_each(num_neurons, vector < vector<double> > (num_state_variables_per_neuron, vector<double>(save_slots,0)));   // Distance from current state of each state variable over sim time.
    #endif
    vector<double> dist_state                (save_slots,0); // Greatest distance from current state of any state variable over sim time.
    vector<double> return_to_state_score_hist(save_slots,0); // How close are we to matching a previous state in the system?
    vector<double> total_dist_hist           (save_slots,0); // Alternate measure of return to state score, more accurate, but slower. Total distance in state space between current state and state at end of simulation. (Calculated as true distance; d^2 = a^2 + b^2 + ...)
    vector<double> edge_dist_hist            (save_slots,0); // Alternate measure of return to state score. Distance as the sum of distances along each dimension. Less accurate than total_dist, but more computationally efficient. (d = a + b + ...)
    vector<vector<vector<double> > > I_hist(num_neurons, vector < vector<double> > (num_channels+4, vector<double>(save_slots,0)));
    //vector<vector<vector<double> > > g_hist(num_neurons, vector < vector<double> > (num_channels,   vector<double>(save_slots,0)));
    //vector<vector<double> > E_Ca_hist  (num_neurons, vector<double>(save_slots,0));
    //vector<vector<double> > A_hist     (num_neurons, vector<double>(save_slots,0));
    
    // Times of spike and burst onsets and offsets.
    vector <double> AB_burst_onsets, PD_burst_onsets, AB_burst_offsets, PD_burst_offsets, AB_spike_onsets, PD_spike_onsets, AB_spike_offsets, PD_spike_offsets;
        
   
    double V_min[] = {+999999.9, +999999.9}; // [Volts] (AB, PD) Min voltage reached in sim, after stabilization period. 
    double V_max[] = {-999999.9, -999999.9}; // [Volts] (AB, PD) Max ...
 
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // Calculate temperature-dependent reversal potentials.
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if (temperature_dependent_reversal_potentials){
        // First find relative ion concentrations for Na+ and K+ inside and outside of the cell, from reversal potential using Nernst equation.
        double K_in  = 400;  // [mMol] Internal K+  concentration. NOTE: Squid values from Kandel. Find crab values to improve accuracy.
        double Na_in =  50;  // [mMol] Internal Na+ concentration
        
        double Na_out = exp(E_Na*1*F/R/reference_Temp)* Na_in; //E = R*T/(z*F)*log(out/in)
        double K_out  = exp(E_K *1*F/R/reference_Temp)* K_in;
        //cout << "Na_out=" << Na_out << " K_out=" << K_out << endl;
        
        // Mixed cation channels.
        // First find relative permiability for mixed cation channel to K+ and Na+. (Other +1 cations are rare and not included in calculation)
        // Permiability can be found by comparing the relative influence of the two ion species: Permiability_channel_2 = (E_rev_mixed_channel - E_rev_1)/(E_rev_2-E_rev_1)
        double E_rev_1 = E_K;      // [V] -80 mV
        double E_rev_2 = E_Na;     // [V] +50 mV
        
        double E_rev_H    = -0.020;  // [V] H    channel reversal potential at reference temperature.    
        double E_rev_proc =  0;      // [V] proc channel reversal potential at reference temperature.
        double P_H_Na     = (E_rev_H    - E_rev_1)/(E_rev_2-E_rev_1 );  // Relative permiability of H    channel to Na+.
        double P_proc_Na  = (E_rev_proc - E_rev_1)/(E_rev_2-E_rev_1 );  // Relative permiability of proc channel to Na+.
        double P_H_K      = 1- P_H_Na;                                  // Relative permiability of H    channel to K+.
        double P_proc_K   = 1- P_proc_Na;                               // Relative permiability of proc channel to K+.
        //cout << "P_H_Na=" << P_H_Na << " P_H_K=" << P_H_K << " P_proc_Na=" << P_proc_Na << " P_proc_K=" << P_proc_Na << endl;
        
        // Single cation channels
        E_Na   = R*T/F*log(Na_out/Na_in); // [Volts] Just Nernst.
        E_K    = R*T/F*log(K_out /K_in ); // [Volts]

        E_H    = P_H_K    * E_K  +  P_H_Na    * E_Na; // Mixed cation reversals are sum of their influences. See Oct 19 notes for derivation, but basically reversal occurs when K+ and Na+ currents are equal and opposite.
        E_proc = P_proc_K * E_K  +  P_proc_Na * E_Na;
        
        //cout << "E_Na= " << E_Na << " E_K=" << E_K << " E_proc=" << E_proc << " E_H=" << E_H << endl;
        
        //cout << "T=" << T-273.15 << " E_H=" << E_H << " E_proc=" << E_proc << " E_Na=" << E_Na << " E_K=" << E_K << endl;
    }
    // Now fill in the rest of the K+ channels.
    //        Na    K_AIS   CaT    CaS  Nap   H    K_Soma  KCa   A    proc
   // E_all = {{E_Na, E_K,    E_Ca, E_Ca, E_Na, E_H, E_K,    E_K,  E_K, E_proc},  // AB [V] Store reversal potentials in vector which matches the channel vector. (Ca values are placeholders.)
   //          {E_Na, E_K,    E_Ca, E_Ca, E_Na, E_H, E_K,    E_K,  E_K, E_proc}}; // PD
    
    E_all[AB][0] = E_Na;// Alternate version because compiler on cluster complains about above version.
    E_all[AB][1] = E_K;
    E_all[AB][2] = E_Ca;
    E_all[AB][3] = E_Ca;
    E_all[AB][4] = E_Na;
    E_all[AB][5] = E_H;
    E_all[AB][6] = E_K;
    E_all[AB][7] = E_K;
    E_all[AB][8] = E_K;
    E_all[AB][9] = E_proc;
    
    E_all[PD][0] = E_Na;
    E_all[PD][1] = E_K;
    E_all[PD][2] = E_Ca;
    E_all[PD][3] = E_Ca;
    E_all[PD][4] = E_Na;
    E_all[PD][5] = E_H;
    E_all[PD][6] = E_K;
    E_all[PD][7] = E_K;
    E_all[PD][8] = E_K;
    E_all[PD][9] = E_proc;
    
    //%%%%%%%%%%%%%%%%
    //   Main Loop   %
    //%%%%%%%%%%%%%%%%

    sim_time  = 0; // [seconds] Total simulation time.
    double stabilization_time = 15.0; // [seconds] Amount of time to run the simulation before starting to record statistics.
    bool step_is_small_enough = true; 
    double step_dt = dt; //[seconds] Starting value for time step.
    while ( sim_time <= (sim_length - dt_hist) ){ // Do simulation until time limit reached.
        if (!cluster) { cout << "sim_time=" << sim_time << "   sim_length=" << sim_length << endl;} 
        
        double epoch_time = 0; //[ms] Time elapsed during current epoch.
        while (epoch_time < epoch_length){
            oneStep(0, 1, step_dt  ); // Do a full step.      (Current state, next state, dt) For Half step method: Step 1 is one full step. Step 2 is first half step. Step 3 is second half step. 0 is always current state.
            oneStep(0, 2, step_dt/2); // Do first half step.
            oneStep(2, 3, step_dt/2); // Do next half step.

            // Check if state from full step is close enough to two half steps. This FIRST VERSION will just test voltage of all 4 compartments.
            step_is_small_enough = true; // NOTE: The test is quite unsophisticated, but works well enough.
            for (int neuron=0; neuron<num_neurons; ++neuron){
                for (int compartment = 0; compartment<num_compartments; ++compartment){
                    if (fabs(voltages[neuron][compartment][1] - voltages[neuron][compartment][3]) > 0.0000002){ //NOTE: Testing scaling with constant Q10s, so using tighter tolerance. Was 0.0000002){ // See July 11 notes for tests of values of this voltage threshold. //step_dt > 0.01 ) {
                        step_is_small_enough = false;
                    }
                }
            }
            
            if (step_is_small_enough){             // Record history, interpolating as needed. // NOTE: Fancier version would make use of first half step for more accuracy.
                while //( 
                    (time_step_hist*dt_hist < sim_time // How far we are along in recording history <  How far along the simulation is.
                    && 
                    sim_time < sim_length)            
                    { // Simplified test.
                    double time_to_record = ((double)time_step_hist+1)*dt_hist;
                    double time_since_last_sim_step = time_to_record - sim_time;
                    double time_step_fraction = time_since_last_sim_step/step_dt;
                    
                    for (int neuron=0; neuron<num_neurons; ++neuron){
                                                
                        for (int compartment = 0; compartment<num_compartments; ++compartment){
                            double last_voltage = voltages[neuron][compartment][0]; // [0] and [3] are step IDs for current state and second half step.
                            double next_voltage = voltages[neuron][compartment][3];
                            V_hist[neuron][compartment][time_step_hist] = last_voltage + (next_voltage - last_voltage)*time_step_fraction; // Interpolation step.

                            #ifdef testing
                                if (isnan((float)V_hist[neuron][compartment][time_step_hist])){
                                    cout << "Error: Voltage is NaN. ----- " << endl << "last_voltage="<<last_voltage<<" next_voltage="<<next_voltage<<" time_step_fraction="<<time_step_fraction<<endl<<"Exiting"<<endl;
                                    exit(-1); 
                                }
                            #endif
                            state_hist[neuron][compartment][time_step_hist] = V_hist[neuron][compartment][time_step_hist]; // Record voltage of each compartment as the first state variables per neuron.
                        }
                       // Removed history recording code from cluster.
                       #ifdef desktop
                            for (int channel = 0; channel < num_channels; ++channel){
                                double last_act   = activations  [neuron][channel][0];
                                double next_act   = activations  [neuron][channel][3];
                                double last_inact = inactivations[neuron][channel][0];
                                double next_inact = inactivations[neuron][channel][3];
                                double last_I     = currents     [neuron][channel][0];
                                double next_I     = currents     [neuron][channel][3];
                                
                                m_hist[neuron][channel][time_step_hist] = last_act   + (next_act   - last_act  )*time_step_fraction;
                                h_hist[neuron][channel][time_step_hist] = last_inact + (next_inact - last_inact)*time_step_fraction;
                                I_hist[neuron][channel][time_step_hist] = last_I     + (next_I     - last_I    )*time_step_fraction;
                                state_hist[neuron][num_compartments +                channel][time_step_hist] = m_hist[neuron][channel][time_step_hist]; // num_compartments offset is for the voltage save slots.
                                state_hist[neuron][num_compartments + num_channels + channel][time_step_hist] = h_hist[neuron][channel][time_step_hist]; // num_channels offset is for the activation (m) save slots.
                            }
                        #endif
                        
                        double last_Ca = Ca[neuron][0];
                        double next_Ca = Ca[neuron][3];
                        Ca_hist[neuron][time_step_hist] = last_Ca + (next_Ca - last_Ca)*time_step_fraction;
                        state_hist[neuron][num_compartments + num_channels*2][time_step_hist] = Ca_hist[neuron][time_step_hist];

                    }
                    ++time_step_hist;
                    //if (time_step_hist >= 40001) {cout << "time_step_hist = "<< time_step_hist <<"   sim_time = " << sim_time << " save_slots = " << save_slots << "  dt_hist = " << dt_hist << "  step_dt = " << step_dt << endl;}
                }

                //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                // Do spike and burst detection %
                //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if (sim_time > stabilization_time){ // Only do spike and burst detection after specified simulation time.
                    for (int neuron = 0; neuron < num_neurons; ++neuron){

                    // Burst detection
                    if (bursting[neuron] ){
                        if (activations[neuron][A][3] < 0.1) {//0.07 // Check activation of A channel. (Value increased to catch borderline cases.)
                            bursting[neuron] = false;
                            // Record burst offset
                            if (neuron == AB){
                                AB_burst_offsets.push_back(sim_time);
                            } else {
                                PD_burst_offsets.push_back(sim_time);
                            }
                        }
                    } else {
                        if (activations[neuron][A][3] > 0.15){ // 0.1
                            bursting[neuron] = true;
                            // Record bursting onset
                            if (neuron == AB) {
                                AB_burst_onsets.push_back(sim_time);
                            } else {
                                PD_burst_onsets.push_back(sim_time);
                            }
                        }
                    }
                    
                    // Spike detection
                    if (spiking[neuron]) {
                        if (activations[neuron][Na][3] < 0.3) // Check activation of Na channel.
                            spiking[neuron] = false;
                        // Record spike offset
                        if (neuron == AB) {
                            AB_spike_offsets.push_back(sim_time);
                        } else {
                            PD_spike_offsets.push_back(sim_time);
                        }
                    } else {
                        if (activations[neuron][Na][3] > 0.7) {
                            spiking[neuron] = true;
                            // Record spike onset
                            if (neuron == AB) {
                                AB_spike_onsets.push_back(sim_time);
                            } else {
                                PD_spike_onsets.push_back(sim_time);
                            }
                        }
                    }
                } // End spike & burst detection.
                }

                // Copy future state (following two half steps) to current state
                for(int neuron = 0; neuron<num_neurons; ++neuron){
                    Ca[neuron][0] = Ca[neuron][3];
                    for (int channel=0; channel<num_channels; ++channel){
                        activations   [neuron][channel][0] = activations   [neuron][channel][3];
                        inactivations [neuron][channel][0] = inactivations [neuron][channel][3];
                        currents      [neuron][channel][0] = currents      [neuron][channel][3];
                    }
                    for(int compartment=0; compartment<num_compartments; ++compartment){
                        voltages[neuron][compartment][0] = voltages[neuron][compartment][3];
                    }

                    V_min[neuron] =  ( voltages[neuron][Soma][0] < V_min[neuron]) ? voltages[neuron][Soma][0] : V_min[neuron];
                    V_max[neuron] =  ( voltages[neuron][Soma][0] > V_max[neuron]) ? voltages[neuron][Soma][0] : V_max[neuron];
                }
                sim_time   += step_dt;
                epoch_time += step_dt;
                
                step_dt = step_dt*1.01; // Increase dt size
            }else{
                step_dt = step_dt/1.8; // Reduce dt and try again.
            }
            //
        }
        //if (!cluster) {cout << "time_step_hist = "<< time_step_hist <<"   sim_time = " << sim_time << " save_slots = " << save_slots << "  dt_hist = " << dt_hist << "  step_dt = " << step_dt << endl;}
        
        
    }
    if (!cluster){cout << "Time simulated was: " << sim_time << endl;}
    //%%%%%%%%%%%%%%%%%
    // End Main Loop %%
    //%%%%%%%%%%%%%%%%%    
    
    struct neuron_stats AB_stats, PD_stats; // This is where the summary data for the neurons lives.
    
    // Remove burst onsets which lack matching offset, if any. (There are never offsets lacking onsets, since we don't look for an offset if an onset has not yet happened.)
    AB_burst_onsets.resize(AB_burst_offsets.size());
    PD_burst_onsets.resize(PD_burst_offsets.size());
    
    // Find the burst lengths. 
    vector <double> AB_burst_lengths;
    vector <double> PD_burst_lengths;
    if(AB_burst_onsets.size() >=1 ){
        AB_burst_lengths.resize(AB_burst_onsets.size());
        for (unsigned int i = 0; i < AB_burst_offsets.size(); ++i){
            AB_burst_lengths[i] = AB_burst_offsets[i] - AB_burst_onsets[i];
        }
    }
    if(PD_burst_onsets.size() >=1 ){
        PD_burst_lengths.resize(PD_burst_onsets.size());
        for (unsigned int i = 0;  i < PD_burst_offsets.size(); ++i){
            PD_burst_lengths[i] = PD_burst_offsets[i] - PD_burst_onsets[i];
        }
    }

    // Find the burst intervals and frequencies.
    vector <double> AB_burst_intervals, PD_burst_intervals;
    if (AB_burst_onsets.size() >= 2) {
        AB_burst_intervals.resize(AB_burst_onsets.size() -1);
        for (unsigned int i = 0; i+1 < AB_burst_onsets.size(); ++i){
            AB_burst_intervals[i] = AB_burst_onsets[i+1] - AB_burst_onsets[i];
        }
        double mean_AB_burst_interval = my_mean(AB_burst_intervals,0.0);
        // Check for regular bursting by computing coefficient of variation of bursting.
        if (AB_burst_onsets.size() >= 3 &&  standard_deviation(AB_burst_intervals)/mean_AB_burst_interval < 0.05) {
            AB_stats.regular_bursting = true;
            AB_stats.mean_burst_frequency = 1/mean_AB_burst_interval; 
        }else{
            AB_stats.mean_burst_frequency = -1; // Don't compute a burst frequency for an irregular burster. 
            // There are many cases where this calculation could give a nonnesical result unless simulation is very long.
        }
    } else {
        AB_burst_intervals.resize(0);
        AB_stats.mean_burst_frequency = -1;
    }
    
    if (PD_burst_onsets.size() >= 2 ) {
        PD_burst_intervals.resize(PD_burst_onsets.size() -1);
        for (unsigned int i = 0; i+1 < PD_burst_onsets.size(); ++i){
            PD_burst_intervals[i] = PD_burst_onsets[i+1] - PD_burst_onsets[i]; // interval is time of next - time of this.
        }
        double mean_PD_burst_interval = my_mean(PD_burst_intervals,0.0);
        if (PD_burst_onsets.size() >= 3 &&  standard_deviation(PD_burst_intervals)/mean_PD_burst_interval < 0.05) {   
            PD_stats.regular_bursting = true;
            PD_stats.mean_burst_frequency = 1/(my_mean(PD_burst_intervals, 0.0));
        }else{
            PD_stats.mean_burst_frequency = -1;
        }
    } else {
        PD_burst_intervals.resize(0);
        PD_stats.mean_burst_frequency = -1;
    }

    long number_of_AB_bursts = AB_burst_offsets.size();
    long number_of_PD_bursts = PD_burst_offsets.size();
    
    // Get index of first spike in first burst and last spike in last whole burst in order to find mean spikes per burst. (There may be spikes in bursts that don't finish at the end.)
    long first_AB_spike = 0;
    long last_AB_spike  = 0;
    bool done = false;
    if (AB_burst_onsets.size() >= 1 && AB_burst_offsets.size() >= 1 ){ // Only test in cases where there is a complete burst.
        for (unsigned int x= 1; x <= AB_spike_onsets.size() && !done; ++x){
            last_AB_spike = x;
            if (AB_spike_onsets[x] > AB_burst_onsets[0] && first_AB_spike == 0){ // Look for a spike which occurs after the time for the first burst onset.
                first_AB_spike = x;
            }
            if (AB_spike_onsets[x] > AB_burst_offsets[AB_burst_offsets.size()-1]){ // If this spike is past the time of the last offset, then we're done.
                done = true;
            }
        }
    }
    long first_PD_spike = 0;
    long last_PD_spike  = 0;
    done = false;
    if (PD_burst_onsets.size() >= 1 && PD_burst_offsets.size() >= 1 ){
        for (unsigned int x= 1; x <= PD_spike_onsets.size() && !done; ++x){
            last_PD_spike = x;
            if (PD_spike_onsets[x] > PD_burst_onsets[0] && first_PD_spike == 0){
                first_PD_spike = x;
            }
            if (PD_spike_onsets[x] > PD_burst_offsets[PD_burst_offsets.size()-1]){
                done = true;
            }
        }
    }

    AB_stats.mean_burst_length = my_mean(AB_burst_lengths, 0);
    PD_stats.mean_burst_length = my_mean(PD_burst_lengths, 0);
    
    // Find spikes per burst.
    if (number_of_AB_bursts){
        AB_stats.mean_spikes_per_burst = (double)last_AB_spike/((double)number_of_AB_bursts);
    }else{
        AB_stats.mean_spikes_per_burst = -1;
    }
    
    if (number_of_PD_bursts){
        PD_stats.mean_spikes_per_burst = (double)last_PD_spike/((double)number_of_PD_bursts);
    }else{
        PD_stats.mean_spikes_per_burst = -1;
    }
    
    // Find duty cycle.  
    if ( AB_stats.regular_bursting == true){
        AB_stats.duty_cycle = AB_stats.mean_burst_length * AB_stats.mean_burst_frequency;
    }else{
        AB_stats.duty_cycle = -1; // Just don't calculate duty cycle for irregular bursters...
    }
    // ... Instead calcualtate a new stat that makes no claim to represtent the true duty cycle;
    // it just calculates what fraction of the time the cell was bursting. 
    double AB_bursting_time = accumulate(AB_burst_lengths.begin(), AB_burst_lengths.end(), 0);
    AB_stats.fraction_bursting = AB_bursting_time/(sim_time - stabilization_time);

    if ( PD_stats.regular_bursting == true){
        PD_stats.duty_cycle = PD_stats.mean_burst_length * PD_stats.mean_burst_frequency;
    }else{
        PD_stats.duty_cycle = -1;
    }
    double PD_bursting_time = accumulate(PD_burst_lengths.begin(), PD_burst_lengths.end(), 0);
    PD_stats.fraction_bursting =  PD_bursting_time/(sim_time - stabilization_time);
    
    
    // Find first and last spike in each burst to get a spike-based duty cycle definition.
    vector <double> AB_bursts_on;
    vector <double> AB_bursts_off;
    vector <double> PD_bursts_on;
    vector <double> PD_bursts_off;
    vector <double> spikes_in_burst; // Times of spikes in a burst.
    for (unsigned int burst=0; burst<AB_burst_onsets.size(); ++burst){
        // Get all spikes within AB bursts.
        spikes_in_burst.clear();
        for ( unsigned int spike_index=0; spike_index<AB_spike_onsets.size(); ++spike_index){
            if (   AB_spike_onsets[spike_index] > AB_burst_onsets [burst]   //     Spike is after burst onset ...
                && AB_spike_onsets[spike_index] < AB_burst_offsets[burst])  // and spike is before burst offset.
                { spikes_in_burst.push_back(AB_spike_onsets[spike_index]);} // Add spike to set.
        }
        if(spikes_in_burst.size() > 1){ // Ignore one spike bursters, since we can't define their duty cycle.
            AB_bursts_on.push_back( spikes_in_burst.front()); // Record time of onset.
            AB_bursts_off.push_back(spikes_in_burst.back() ); // Record time of offset.
        }
    }
    for (unsigned int burst=0; burst<PD_burst_onsets.size(); ++burst){ 
        // Get all spikes within PD bursts.
        spikes_in_burst.clear();
        for (unsigned int spike_index=0; spike_index<PD_spike_onsets.size(); ++spike_index){
            if (   PD_spike_onsets[spike_index] > PD_burst_onsets [burst]   //     Spike is after burst onset ...
                && PD_spike_onsets[spike_index] < PD_burst_offsets[burst])  // and spike is before burst offset.
                { spikes_in_burst.push_back(PD_spike_onsets[spike_index]);} // Add spike to set.
        }
        if(spikes_in_burst.size() > 1){ // Ignore one spike bursters, since we can't define their duty cycle.
            PD_bursts_on.push_back( spikes_in_burst.front()); // Record time of onset.
            PD_bursts_off.push_back(spikes_in_burst.back() ); // Record time of offset.
        }
    }
    
    // Calculate duty cycle by the new spikes in burst method.
    double total_AB_burst_time = 0; // Time spent within a burst. (Single burst time = time of last spike - time of first spike)
    double total_PD_burst_time = 0;
    //cout << "AB burst times = ";
    for (unsigned int burst=0; burst<AB_bursts_on.size(); ++burst){total_AB_burst_time += AB_bursts_off[burst] - AB_bursts_on[burst]; cout << AB_bursts_off[burst] - AB_bursts_on[burst] << "  ";} // Find total time spent within bursts.
    cout << endl;
    for (unsigned int burst=0; burst<PD_bursts_on.size(); ++burst){total_PD_burst_time += PD_bursts_off[burst] - PD_bursts_on[burst];}
    //cout << "total_AB_burst_time=" << total_AB_burst_time << "  total_PD_burst_time=" << total_PD_burst_time << endl;
    AB_stats.duty_cycle_by_spikes = 0; // Default duty cycle = 0.
    PD_stats.duty_cycle_by_spikes = 0;
    if (AB_bursts_on.size() > 1) {
        double AB_burst_time_basline = AB_bursts_on.back() - AB_bursts_on[0]; 
        //cout << "total AB time=" << AB_burst_time_basline << endl;
        AB_stats.duty_cycle_by_spikes = total_AB_burst_time/AB_burst_time_basline;
    } // If there is enough data, divide bursting time by total time to get duty cycle.
    if (PD_bursts_on.size() > 1) {
        double PD_burst_time_basline = PD_bursts_on.back() - PD_bursts_on[0];
        //cout << "total PD time=" << PD_burst_time_basline << endl;
        PD_stats.duty_cycle_by_spikes = total_PD_burst_time/PD_burst_time_basline;
    }
    //cout << "AB_stats.duty_cycle_by_spikes=" << AB_stats.duty_cycle_by_spikes << "  PD_stats.duty_cycle_by_spikes=" << PD_stats.duty_cycle_by_spikes << endl;

    AB_stats.V_min = V_min[AB];   
    AB_stats.V_max = V_max[AB];   

    PD_stats.V_min = V_min[PD];   
    PD_stats.V_max = V_max[PD];   

    // Set up all the data to return. 
    struct network_stats the_network_stats = {
        AB_stats, PD_stats, save_slots,
        V_hist,
        #ifdef desktop
            I_hist, m_hist, h_hist, Ca_hist,
        #endif
        AB_burst_onsets,  // Classified by activation of A current.
        PD_burst_onsets,
        AB_burst_offsets,
        PD_burst_offsets,
        AB_spike_onsets,
        PD_spike_onsets,
        AB_bursts_on,    // Classified by spike times within burst. (Note difference from burst_onsets.)
        AB_bursts_off,
        PD_bursts_on,
        PD_bursts_off,

        g_max_all_adj, // Maximal channel conductances     adjusted for Q10 values.
        g_max_all,     // Maximal channel conductances NOT adjusted for Q10 values.
        g_axial,
        g_gap,
        g_leaks
    };
    
    return (the_network_stats);

} // End function: model().

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//             Simulate one time step.                        %
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void oneStep(int last_step, int next_step, double dt){
    
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // Find channel activation and inactivations  %
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for (int neuron = 0; neuron <num_neurons; ++neuron){
        for (int channel = 0; channel < num_channels; ++channel) {
            int compartment = -1;
            if (channel < 2) {compartment = AIS;} else {compartment = Soma;} // First two channels are in axon initial segment.
            double V = voltages[neuron][compartment][last_step];  // [V]
            #ifdef testing
                if (isnan((float)V)){cout << "Voltage is NaN. Exiting."<<endl; exit(-1);}
            #endif
            double Q_m_alpha = Q_alphas[ 0][ channel]; // 0 is activation, 1 is inactivation. 
            double Q_h_alpha = Q_alphas[ 1][ channel];
            double Q_m_beta  = Q_betas [ 0][ channel];
            double Q_h_beta  = Q_betas [ 1][ channel];
            double m=0, h=0;
            #ifdef testing   // Turn off after testing for better performance.
                int error = get_channel_state(neuron, channel, V, Ca[neuron][last_step], activations[neuron][channel][last_step], inactivations[neuron][channel][last_step], dt, Q_m_alpha, Q_h_alpha, Q_m_beta, Q_h_beta, &m, &h);
                if (error != 0){ cout << "Error in get_channel_state function." << endl; exit(-1); }
                if (m <0.0 || m > 1.01 || h < 0.0 || h > 1.01 || isnan((float)m) || isnan((float)h)){
                    cout << "Activation or inactivation out of range. --- h=" << h << "  m=" << m << endl;
                    exit(-1);
                }
            #else
                get_channel_state(neuron, channel, V, Ca[neuron][last_step], activations[neuron][channel][last_step], inactivations[neuron][channel][last_step], dt, Q_m_alpha, Q_h_alpha, Q_m_beta, Q_h_beta, &m, &h);
            #endif
            activations  [neuron][channel][next_step]            = m;
            inactivations[neuron][channel][next_step]            = h;
        }
    }
    
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // Calculate channel conductances  %
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    // Find channel conductances
    for (int neuron = 0; neuron < num_neurons; ++neuron){
        // Find Ca reversal potential via Nernst equation.
        double Ca_in = Ca[neuron][last_step];
        if (temperature_dependent_reversal_potentials){
            E_Ca = R*T/(z*F)*log(Ca_out/Ca_in); // [Volts]
        } else {
            E_Ca = R*reference_Temp/(z*F)*log(Ca_out/Ca_in); // [Volts]
        }
        if (constant_Ca_reversal_potential){ // Used only for testing model.
            E_Ca = R*reference_Temp/(z*F)*log(Ca_out/Ca_steady_state); // [Volts]
        }
        E_all[neuron][ CaT] = E_Ca; // Set reversal values for Ca++.
        E_all[neuron][ CaS] = E_Ca;
        
        for (int channel = 0; channel < num_channels; ++channel) {
            double m = activations[neuron][channel][last_step];
            double h = inactivations[neuron][channel][last_step];
            double g_max = g_max_all_adj[neuron][channel]; 
            
            double a = a_exponents[neuron][channel];
            double b = b_exponents[channel];
            if (use_Liu_CaS && (channel == CaS)) { // The Liu CaS channel model includes inactivation, which the standard model doesn't...
                b=1; // ... so include the inactivation in conductance calculation.
            }
            
            g_channels[neuron][ channel] = g_max * pow(m, a) * pow(h, b); // g_max * m^a * h^b;
            #ifdef testing   // Turn off after testing for better performance.
                double g_channel = g_channels[neuron][channel];
                if ( (g_channel < 0) ) {// Testing for imaginary or negative component.
                    cout << "Negative conductance. Stopping." << endl;
                    exit(-1);
                }
            #endif
        }
    }
    
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // Find new voltage of each compartment %      
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    // Find coupling current
    double dV_neurons = voltages[PD][Soma][last_step]  - voltages[AB][Soma][last_step];
    double coupling_current =  dV_neurons * g_gap_adj; // NOTE: *Positive* current if PD is at higher potential. (Current flows into cell AB.)
    
    // Find currents within each neuron
    for (int neuron = 0; neuron<num_neurons; ++neuron){
        // Sum the currents for each compartment
        double dV_axial = voltages[neuron][Soma][last_step] - voltages[neuron][AIS][last_step]; // NOTE: Positive for current flowing into AIS from soma.
        double axial_current   = dV_axial * g_axial_adj[neuron];
        short  coupling_current_direction = ((neuron==AB)*2 -1); // +1 for cell 1 (AB). -1 for cell 2 (PD).
        
        double I_weighted_sum  [num_compartments]; // Sum of single steady state channel currents gives weighted sum of their contributions at steady state.
        double current_sum     [num_compartments]; // Sum of all currents     (active, leak, axial, gap and applied).
        double conductance_sum [num_compartments]; // Sum of all conductances (active, leak, axial and gap)
        
        for (int compartment = 0; compartment < num_compartments;  ++compartment){
            I_weighted_sum[compartment]  = 0;
            current_sum[compartment]     = 0;
            conductance_sum[compartment] = 0;
            double V = voltages[neuron][compartment][last_step];
            int first_channel, last_channel, I_axial_direction;
            if (compartment == AIS) {first_channel = Na; last_channel = K_AIS;} else {first_channel = CaT; last_channel = proc;} 
            if (compartment == AIS) {I_axial_direction = 1;}                    else {I_axial_direction = -1; } // Axial current direction is positive for current flowing into AIS.
                for (int channel = first_channel; channel<= last_channel; ++channel) {// Iterate through the set of channels in the compartment.
                    
                    double channel_current = g_channels[neuron][ channel]* (E_all[neuron][channel] -V);
                    currents[neuron][channel][next_step] = channel_current;
                    if (channel == CaT ) {I_CaT[neuron] = channel_current;}
                    if (channel == CaS ) {I_CaS[neuron] = channel_current;}
                    current_sum[compartment] += channel_current;
                    double I_weight        = g_channels[neuron][ channel]*  E_all[neuron][ channel]; // Single channel current weight.
                    I_weighted_sum[compartment]  += I_weight;
                    conductance_sum[compartment] += g_channels[neuron][ channel];
                }
                
                double I_leak_weight  =  g_leaks_adj[neuron][compartment]*(E_leak[neuron][compartment]   ) ;
                double leak_current   =  g_leaks_adj[neuron][compartment]*(E_leak[neuron][compartment] -V) ;
                if (compartment == Soma){
                    int other_neuron;
                    if (neuron == AB) {other_neuron = PD;} else {other_neuron = AB;}
                    double I_axial_weight = g_axial_adj[neuron]*voltages[neuron][AIS][last_step];
                    double I_gap_weight   = g_gap_adj * voltages[other_neuron][Soma][last_step];
                    conductance_sum[Soma] +=  g_leaks_adj[neuron][Soma] + g_axial_adj[neuron] + g_gap_adj;
                    I_weighted_sum[Soma]  +=  I_leak_weight + I_axial_weight + I_gap_weight; // + I_inj[neuron];
                    current_sum[Soma]     +=  leak_current + axial_current*I_axial_direction + coupling_current*coupling_current_direction; // + I_inj[neuron];
                }else{
                    #ifdef testing
                        if (compartment != AIS) {cout << "ERROR: Unknown compartment." << endl; exit(-1); }
                    #endif
                    double I_axial_weight = g_axial_adj[neuron]*voltages[neuron][Soma][last_step];
                    conductance_sum[AIS]  += g_leaks_adj[neuron][ AIS] + g_axial_adj[neuron];
                    I_weighted_sum[AIS]   += I_leak_weight + I_axial_weight;
                    current_sum[AIS]      += leak_current  + axial_current*I_axial_direction;
                }
        }
        
        // Now calculate new voltage from currents.
        for (int compartment = 0; compartment < num_compartments;  ++compartment){
            double V = voltages[neuron][compartment][last_step];
            double V_inf, tau_V;
            V_inf  =  I_weighted_sum[compartment]/conductance_sum[compartment];         // Steady state voltage.
            tau_V  =  capacitance[neuron][compartment]/(conductance_sum[compartment]);  // Membrane time constant.
            voltages[neuron][compartment][next_step] =  V_inf + (V-V_inf)*exp(-dt/tau_V); //
        }
    }
    
    //%%%%%%%%%%%%%%%%%%%%
    // Calculate [Ca++]  %
    //%%%%%%%%%%%%%%%%%%%%
    
    for (int neuron = 0; neuron < num_neurons; ++neuron){
        double I_Ca  = I_CaT[neuron] + I_CaS[neuron];
        
        double Ca_steady_state_adj = Ca_steady_state + I_Ca*F_Ca[neuron]; // Adjusted steady state [Ca++] includes Ca++ current.
        Ca[neuron][next_step] = Ca_steady_state_adj + (Ca[neuron][last_step] - Ca_steady_state_adj)*exp(-dt/(tau_Ca[neuron]/Q_Ca));  // Eponential Euler method: inf + (curr - inf)*exp(-dt/tau)
        if (constant_Ca_concentration) {Ca[neuron][next_step] = Ca_steady_state;} // For testing only to ensure system scales with universal Q10, while eliminating things like [Ca++] changes, which may interfere with this relationship.
        
        //cout << "I_Ca=" << I_Ca << "   Ca_steady_state_adj=" << Ca_steady_state_adj << "  [Ca]=" << Ca[neuron][next_step] << "  Q_Ca=" << Q_Ca << "  dt=" << dt << "  tau_Ca=" << tau_Ca[neuron] << "  Delta_Ca=" << delta_Ca << endl;
        
        #ifdef testing   // Turn off after testing for better performance.
            if (Ca[neuron][next_step] < 0){
                if (sim_time < 10*pow(10.0,-3)){
                    cout << "Negative [Ca++] value early in sim. Resetting." << endl;);
                    Ca[neuron][next_step] = 0;
                }else{
                    cout << "Negative [Ca++] value. Stopping." << endl;
                    exit(-1);
                }
            }
        #endif
    }
}


//%%%%%%%%%%%%%%%%%%%%
// Helper Functions  %
//%%%%%%%%%%%%%%%%%%%%

int get_channel_state(int neuron, int channel, double V, double Calcium, double old_m, double old_h, double dt, double Q_m_alpha, double Q_h_alpha, double Q_m_beta, double Q_h_beta, double * act, double * inact){
    
    V  = V  * 1000.0; // Convert voltage from V to mV for use in functions table. (Elsewhere voltage is always in Volts.)
    Calcium = Calcium * pow(10.0,6.0); // Convert intracellular [Ca++] from Mol to μM. (Elsewhere [Ca++] is always in Mol.)
    double m=-1, h=-1;
    double m_inf, h_inf=1, tau_m, tau_h=1; //Setting values for h_inf and tau_h avoids compiler warning when -O1 optimization is on.

    switch (channel){
        case Na: // I_Na (Axon initial segment)
            m_inf = sigmoid(-1, 24.7, 5.29, V);
            h_inf = sigmoid(+1, 48.9, 5.18, V);
            tau_m = tau_sigmoid(1.32, -1.26, -1,  120, 25, V);
            tau_h = tau_sigmoid(0,     0.67, -1, 62.9, 10, V)  * tau_sigmoid(1.5, 1, +1, 34.9, 3.6, V) * 1000; // NOTE: The *1000 term adjusts for the fact that tau_sigmoid converts ms to sec, but we call it twice here.
            break;

        case CaT: // I_CaT
            m_inf = sigmoid(-1, 25, 7.2, V);
            h_inf = sigmoid(+1, 36, 7  , V);
            tau_m = tau_sigmoid(55, -49.5, -1, 58, 17, V);
            switch (neuron){
                case 0: // AB
                    tau_h = tau_sigmoid(87.5,  -75, -1, 50, 16.9, V);
                    break;
                case 1: // PD
                    tau_h = tau_sigmoid(350,  -300, -1, 50, 16.9, V);
                    break;
                default:
                    cout << "Error: Cannot find neuron." << endl;
                    return(-1);
                    break;
            }
            break;
            
        case CaS: // I_CaS            
            if ( ! use_Liu_CaS){ // Use standard Turrigiano/Soto-Trevino non-inactivating CaS channel.
                m_inf = sigmoid(-1, 22, 8.5, V); // NOTE: These values are for the Soto-Trevino model of the CaS channel. Liu and Prinz add inactivation.
                tau_m = tau_sigmoid(16, -13.1, -1, 25.1, 26.4, V);
            } else{ // Use Liu CaS channel with inactivation. (Liu et al 1998.)
                m_inf = sigmoid( 1, 33, -8.1, V);
                h_inf = sigmoid( 1, 60,  6.2, V);
                double mV = V/1000; // Convert voltage to proper units for equations below.
                tau_m = (1.4  + (  7.0/(  exp((mV + 27.0)/10.0) + exp((mV + 70.0)/(-13))  )) ) * 0.001;
                tau_h = (60.0 + (150.0/(  exp((mV + 55.0)/ 9.0) + exp((mV + 65.0)/(-16))  )) ) * 0.001;
                //cout << "Using Liu CaS channel. m_inf = "<< m_inf <<"     h_inf = "<< h_inf <<"  tau_m = "<< tau_m <<"  tau_h = "<< tau_h << endl;
            }
            break;
            
        case nap: // I_NaP
            m_inf = sigmoid(-1, 26.8, 8.2, V);
            h_inf = sigmoid( 1, 48.5, 4.8, V);
            tau_m = tau_sigmoid(19.8, -10.7, -1, 26.5,  8.6,  V);
            tau_h = tau_sigmoid(666,  -379,  -1, 33.6,  11.7, V);
            break;
            
        case H: // I_h
            m_inf = sigmoid( 1, 70, 6, V);
            tau_m = tau_sigmoid(272, +1499, -1, 42.2, 8.73, V);
            break;
            
        case K_AIS: case K_soma: // I_K {axon initial segment, soma}
            m_inf = sigmoid(-1, 14.2, 11.8, V);
            tau_m = tau_sigmoid( 7.2, -6.4, -1, 28.3, 19.2, V);
            break;
            
        case KCa: // I_KCa
            switch (neuron){
                case 0:
                    m_inf = Calcium/(Calcium + 30.0) * sigmoid(-1, 51, 4, V);
                    break;
                case 1:
                    m_inf = Calcium/(Calcium + 30.0) * sigmoid(-1, 51, 8, V);
                    break;
                default:
                    cout << "Error: Cannot find neuron." << endl;
                    return(-1);
                    break;
            }
            tau_m = tau_sigmoid(90.3, -75.09, -1, 46, 22.7, V);
            break;
            
        case A: // I_A 
            m_inf = sigmoid( -1, 27,   8.7,  V);
            h_inf = sigmoid( +1, 56.9, 4.9,  V);
            tau_m = tau_sigmoid( 11.6, -10.4, -1, 32.9, 15.2,  V);
            tau_h = tau_sigmoid( 38.6, -29.2, -1, 38.9, 26.5,  V);
            break;
            
        case proc: // I_proc
            m_inf = sigmoid( -1, 12, 3.05,   V);
            tau_m = 0.5*(0.001); //[seconds]
            break;
            
        default:
            cout << "Error: Channel not found. Cannot calculate activation." << endl;
            return(-1);
            break;
    }
    //cout << "m_inf = " << m_inf << "  h_inf="<< h_inf << endl;
    
//     ************** The Q10's get incorporated here ************
//     // Calculate/re-express alphas and betas and incorporating Q10s
//     // (Using m_inf and tau values from above for each channel) 
//     // General form: 
//     //               alpha = m_inf/tau
//     //               beta = (1-m_inf)/tau
//     //(Only necessary for applying Q10s to alphas and betas.)
    
    double m_alpha = Q_m_alpha * m_inf/tau_m;         // Gate opening rate = rate multiplier * (gate steady state / time constant). (If m_inf=0, no channels will open; for m_inf=1, channels open at max rate.)
    double m_beta  = Q_m_beta  * (1.0-m_inf)/tau_m;   // Gate closing rate.
    //cout << "m_alpha=" << m_alpha << "  m_beta=" << m_beta << "  Q_m_alpha=" << Q_m_alpha << "  Q_m_beta=" << Q_m_beta  << endl;
    
    
//     // Recalculate m_inf from alphas and betas with incorporated Q10's general form: m_inf = alpha/(alpha + beta)
    m_inf = m_alpha/(m_alpha + m_beta);
    
//     // Recalculate taus with incorporated Q10's in terms of alpha & betas.
//     // General form: tau = 1 / (alpha + beta) for a given channel
    tau_m = 1.0 / (m_alpha + m_beta);
    
    switch (channel){
        double h_alpha, h_beta;
        case CaS:
            if ( ! use_Liu_CaS) {break;} // Read this one carefuly. This case falls through to the next if we are using Liu CaS channels.
        case Na: case CaT: case nap: case A:  // Channels which inactivate.
            h_alpha = Q_h_alpha * h_inf/tau_h;
            h_beta  = Q_h_beta  * (1.0-h_inf)/tau_h;
            
            h_inf   = h_alpha/(h_alpha + h_beta);
            
            tau_h   = 1.0 / (h_alpha + h_beta);
            break;
    }
    //cout << "m_inf = " << m_inf << "  h_inf="<< h_inf << "(Aphas and betas included.)" << endl;
    
    //  **************** End of Q10 code ******************************    
    
    // Integrate to find m and h at next time step.
    m = m_inf + (old_m - m_inf)*exp(-dt/tau_m);       // Exponential Euler method: inf + (curr - inf)*exp(-dt/tau);  
    switch (channel){
        case CaS:
            if (use_Liu_CaS){ // Liu's and Prinz's CaS channels inactivate, unlike Turrigiano's and Soto-Trevino's.
                h = h_inf + (old_h - h_inf)*exp(-dt/tau_h);       //inf + (curr - inf)*exp(-dt/tau); 
            }else{ 
                h = 1;
            }
            break;
        case KCa: case K_AIS: case H: case K_soma:  case proc:  // Channels which do not inactivate.
            h = 1;
            break;
        case Na: case CaT: case nap: case A: // Channels which inactivate.
            h = h_inf + (old_h - h_inf)*exp(-dt/tau_h);       //inf + (curr - inf)*exp(-dt/tau); 
            break;
        default:
            cout << "Error: Channel not found. Cannot calculate inactivation." << endl;
            return(-1);
            break;
    }
    #ifdef testing
        if (m < 0 || m > 1 || isnan((float)m)) { cout << "m out of range. Stopping. m=" << m << " Channel = "<< channel <<", m_inf= "<< m_inf <<" m_alpha="<< m_alpha <<" m_beta="<< m_beta <<" V="<< V << endl; return(-1);}
        if (h < 0 || h > 1 || isnan((float)h)) { cout << "h out of range. Stopping. h=" << h << " Channel = "<< channel <<", h_inf= "<< h_inf <<" h_alpha="<< h_alpha <<" h_beta="<< h_beta <<" V="<< V << endl; return(-1);}
    #endif
    *act   = m;
    *inact = h;
    
    return(0);
}

// Sigmoid functions for calculating channel dynamics
double sigmoid(double a, double b, double c, double  V) {// a is direction of curve, -b is half (in)activation voltage, V is votlage.
    return ( 1.0/(1.0+exp(a*(V+b)/c)) );
}

double tau_sigmoid(double a, double b, double c, double d, double e, double V){
    return (a + b/(1.0+exp(c*(V+d)/e)))*(0.001); //[seconds] Note the conversion from ms to seconds here.
}

// Return the mean of a vector, or a default value if the vector is of zero length.
double my_mean (vector <double> input, double default_for_div0){
    if (input.size() == 0) return default_for_div0;
    double mean;
    //mean = std::accumulate(input.begin(), input.end(), 0)/((double) input.size());
    double sum = 0;
    for (unsigned int i=0; i<input.size(); ++i){
        sum += input[i];
    }
    mean = sum/((double)input.size());
    return (mean);  
}

// Return standard deviation
double standard_deviation (vector <double> input){
    double mean = my_mean(input,0);
    double sum = 0;
    for (unsigned int i=0; i<input.size(); ++i){
        sum += pow((input[i] - mean),2);
    }
    return pow(sum/((double )input.size() - 1.0),0.5);
}
