// Calls main model, setting up Q10 parameters first.

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <random>


#include "model.h"

using namespace std;

// Prototypes
int    run_one_sim(double temperature, int sim_number, int num_sims, int work_unit);
void   getQ10values(int sim_number, int num_sims); // Generate next set of Q10 values.
double getQ10_alpha_beta_values(int channel_number, int activation, int sim_number);
void   load_rand_numbers(int num_sims);
void setup_Slice_code();

//////////////
// Globals  //
//////////////


// Data saving parameters
bool cluster                 = false;  // Is the simulation being run from the cluster? (if cluster==true, it eliminates various printed and voluminous saved output.) 
                                       // Note: Command line option overrides this, so leave at default value of false.
bool save_voltage            = false;  // Votage trace of simulation.
bool save_spikes_and_bursts  = true;   // Whether to save spike times and burst times from each sim.
bool save_state_vars         = false;  // State vars are activation and inactivation values
bool save_current            = false;  // Current through each channel.
bool save_Ca                 = false;  // Internal Ca++ concentration.
bool save_RTS_scores         = false;  // RTS is "Return To State" - not currently used.

// Simulation parameters
bool use_same_alpha_and_beta            = true;   // Use same values for opening and closing rate of a channel. 
bool use_same_Q10s_for_both_Kds         = true;   // The Kd channel is located in both axon and some. Ensure they have the same Q10 values.
bool use_preset_Q10_for_select_channels = false;  // Replace random Q10 values with set Q10 values from the literature.
enum Q10_selection_methods {constant, randomize, slice, single_points/*, rerun*/}; // How Q_10s are selected. 
                                                    // "constant" is for testing; 
                                                    // "random" samples entire space; 
                                                    // "slices" (probably 2D) samples sub-spaces, 
                                                    // "rerun" takes data stored in a resultsX.txt file and re-runs it. (Useful for making voltage traces from interesting cases where that was not recorded on cluster.)
int  Q10_selection_method                = randomize; 
bool randomize_Q10_Ca                    = true; // (Only for randomize case.) Select a random value for Ca++ buffering time constant Q10 over the default diffusion Q10 value.
bool randomize_Q10_axial                 = false; // "
bool randomize_Q10_gap                   = false; // "
bool randomize_Q10_leak                  = false; // "

int  identical_Q10s_value               = 5;     // Use this value only if all Q10s are set to be equal.
double single_points_default_value      = 2.0;
const bool constant_Ca                  = false; // Hold internal [Ca++]  constant. Should normally be false. Used for model verification to test for perfect scaling.
int        num_base_sets                = 1;    // The number of base sets of conductances used in this set of sims. Also overridden as optional command line option.

int    job_ID                           = 0;     // ID used in parrallel runs. Set via command line parameter. Used to ensure each sim has correct random numbers. load_rand_numbers().
int    g_max_set                        = 0;     // Which set of max conductances to load from file. "0" indicates use default g_max values. Other values set via command line argument.
bool   randomize_initial_conductances   = false; // Alternative to using fixed base sets of conductances. Should be false, unless seeking new base conductance sets.
                                                 // This randomizes around the conductances selelcted in g_max_set.
double Q10_min       = 1.0;  // Q10_min and Q10_max  are used when Q10's are selected randomly.
double Q10_max       = 4;    //  " 
double diffusion_Q10 = 1.6;

// Variables use in doing sims in slices of Q10 space. 
double Q10_alpha_min [2][num_channels]; // [activation/inactivation][num_channels] Ranges to use for slices of Q10 space. For constrained dimensions, min==max. Note: Currenly apha and beta Q10 values must be equal.
double Q10_alpha_max [2][num_channels]; 
int    num_slice_sample_points = 25; // Dimensions of a slice. e.g. 25x25. Assuming slice is square.
double default_slice_Q10_value = 2; 
double varied_Q10_min = 1;
double varied_Q10_max = 4;
int act_1   = -1; // First free Q10: (in)activation. Used only for slice code.
int cond_1  = -1; // First free Q10 channel number.
int act_2   = -1; // Second ...
int cond_2  = -1; // Second ...


// Model variables
double Q10_alphas [2][num_channels]; // First dimension is activation/inactivation, second dimension is channel.
double Q10_betas  [2][num_channels];
double Q10_g_bars [num_channels];
double Q10_Ca;  // Represents the buffering of Ca++, not the rate of entry of Ca++ into the cell.
double Q10_axial[num_neurons];
double Q10_gap;
double Q10_leak;
const double dt_hist =  0.0001;    // [seconds] Time step for recording simulation history.

double random_doubles[200][num_channels*2+5]; // Numbers to be used in selecting Q10 values. 200 is max sims per job. 25 is number of randoms required per sim. 2 per channel (activation, inactivation Q10) plus 5 others (Q10_Ca, 2*Q10_axial, Q10_gap and Q10_leak)

char save_path           [250]; // Where to save results.
char results_file_name   [250]; // File name for Q10s and burst statistics.

//////////
// Main //
//////////

int main(int argc, char* argv[]){
        
    // Default runtime paramenters. May be altered by command line arguments.
    int    first_work_unit = 0;  // ID of first work unit file to execute. Default to executing single work unit.
    int    last_work_unit  = 0;  // ID of last  work unit file to execute.
    int    num_sims        = 1;  // Number of simulations to run per work unit. Also optionally set by command line argument, below.
    double min_temperature = 7;  // [C]
    double temp_step_size  = 4;  // [C]
    int    num_temp_steps  = 5;  
    
    cout <<  "  -----------------------=================    Main program started.   =================--------------------------" << endl;
   
    // Process command line arguments.
    if (argc > 1){ // If an argument has been passed.
        for (int theArg=1; theArg < argc; ++theArg){
            string x = argv[theArg];
            if (!x.compare("cluster") || !x.compare("-cluster") || !x.compare("--cluster") ) { // compare() returns 0 for match.
                cluster = true; 
            }
            if (!x.compare("min_temperature")) { // Get lowest simulation temperature.
                if (theArg +2 > argc) {cout << "Missing command line argument."<< endl; exit(-1);}
                min_temperature = atof(argv[theArg+1]);                
                cout << "min_temperature=" << min_temperature << endl;
            }
            if (!x.compare("temp_step_size")) { // Get size of temperature intervals.
                if (theArg +2 > argc) {cout << "Missing command line argument."<< endl; exit(-1);}
                temp_step_size = atof(argv[theArg+1]);                
                cout << "temp_step_size=" << temp_step_size << endl;
            }
            if (!x.compare("num_temp_steps")) { // Get number of temperature intervals.
                if (theArg +2 > argc) {cout << "Missing command line argument."<< endl; exit(-1);}
                num_temp_steps = atoi(argv[theArg+1]);                
                cout << "num_temp_steps=" << num_temp_steps << endl;
            }
            if (!x.compare("num_sims")) { // Get number of simulations to do per work_unit.
                if (theArg +2 > argc) {cout << "Missing command line argument."<< endl; exit(-1);}
                num_sims = atoi(argv[theArg+1]);                
                cout << "num_sims=" << num_sims << endl;
                if (num_sims > 200) {cout << "Too many sims per job. Max is 200. (Alternatively, fix segfault at 204 sims.)"; exit (-1);} // Segfault likely fixed via removal of leaky return to state code.
            }
            if (!x.compare("first_work_unit")) { // Get ID of first work_unit..
                if (theArg +2 > argc) {cout << "Missing command line argument."<< endl; exit(-1);}
                first_work_unit = atoi(argv[theArg+1]);                
                cout << "first_work_unit=" << first_work_unit << endl;
            }
            if (!x.compare("last_work_unit")) { // Get ID of last work_unit..
                if (theArg +2 > argc) {cout << "Missing command line argument."<< endl; exit(-1);}
                last_work_unit = atoi(argv[theArg+1]);                
                cout << "last_work_unit=" << last_work_unit << endl;
            }
            if (last_work_unit < first_work_unit) {   // Ensure last work unit is not smaller than first work unit.
                last_work_unit = first_work_unit;
                cout << "Updated: last_work_unit=" << last_work_unit << endl;
            }
            if (!x.compare("g_max_set")) { // Get ID of set of g_max's to use.
                if (theArg +2 > argc) {cout << "Missing command line argument."<< endl; exit(-1);}
                g_max_set = atoi(argv[theArg+1]);                
                cout << "g_max_set=" << g_max_set << endl;
            }
            if (!x.compare("job_ID")) { // Get job_ID this is used to ensure that each sim gets the correct random numbers. See note in load_rand_numbers();
                if (theArg +2 > argc) {cout << "Missing command line argument."<< endl; exit(-1);}
                job_ID = atoi(argv[theArg+1]);                
                cout << "job_ID=" << job_ID << endl;
            }
            if (!x.compare("num_base_sets")) { // Get number of base sets of conductances.
                if (theArg +2 > argc) {cout << "Missing command line argument."<< endl; exit(-1);}
                num_base_sets = atoi(argv[theArg+1]);                
                cout << "num_base_sets=" << num_base_sets << endl;
            }
            if (!x.compare("free_Q10s_for_slice")) { // Get which variables are altered in slices through Q10 space.
                if (theArg +5 > argc) {cout << "Missing command line argument."<< endl; exit(-1);}
                act_1  = atoi(argv[++theArg]);                
                cond_1 = atoi(argv[++theArg]);                
                act_2  = atoi(argv[++theArg]);                
                cond_2 = atoi(argv[++theArg]);            
                cout << "Free Q10s are: act_1 " << act_1 << " cond_1 " << cond_1<< " act_2 " << act_2 << " cond_2 " << cond_2  << endl;
                Q10_selection_method = slice; //Ensure we are selecting Q10s by slice.
            }
            
            if (!x.compare("num_slice_sample_points")) { // Get num_slice_sample_points. (2D grid, this many points in each dimension.)
                if (theArg +2 > argc) {cout << "Missing command line argument."<< endl; exit(-1);}
                num_slice_sample_points = atoi(argv[theArg+1]);                
                cout << "num_slice_sample_points=" << num_slice_sample_points << endl;
            }

            if (!x.compare("randomize_initial_conductances")) { // Get whether to randomize the conductacnes given in g_max_set.
                string value = argv[theArg+1];
                if (theArg +2 > argc) {cout << "Missing command line argument."<< endl; exit(-1);}
                if (! value.compare("true")){
                    randomize_initial_conductances = true;
                }else{
                    if (! value.compare("false")){
                        randomize_initial_conductances = false;
                    } else {
                        cout << "Illegal option for randomize_initial_conductances. Value must be either \"true\" or \"false\"." << endl;
                        exit(-1);
                    }
                }
                cout << "randomize_initial_conductances=" << randomize_initial_conductances << endl;
            }

        }
    }
    
    load_rand_numbers(num_sims); // Set up random number generator by advancing it to the correct spot in the sequence.
    
    // Set up slice code.
    if (Q10_selection_method == slice){
        if (act_1 < 0 || act_2 < 0 || cond_1 < 0 || cond_2 < 0 ) {cout << "Must set up indices for free Q10 variables when doing slices." << endl; exit(-1);}
        setup_Slice_code(); 
    }
    
    if (!cluster){ // Save path for desktop
      // sprintf(save_path, "/home/jon/Documents/Q10_project/model/plot_data/individaul_runs_with_voltage_and_state_data/varied_CaS_Ramp_Q10_KCa=2_standard_CaS_channel_short_dt_corrected_Ca/%d/", g_max_set);
        sprintf(save_path, "./%d/", g_max_set);
    }else{ // Save path for cluster
            sprintf(save_path, "./%d/", g_max_set); // Just save to the local directory (with subfolder for the base conductance set).
    }
    cout << "Save path is: " <<  save_path << endl;
    
    // Create directory for saving results.
    int dir_error = mkdir(save_path, S_IRWXU | S_IRWXG); // Full permissions.
    if (dir_error){ // Check for error.
            cout << "Error creating directory: " << dir_error << endl; // Note: This will throw error on directory exists.
    }
 
    
    for (int work_unit = first_work_unit; work_unit<= last_work_unit; ++work_unit){       // Run set of sims
        
        sprintf(results_file_name,    "%sresults%d.txt",    save_path, work_unit); // Where to save summary of all simulation results.
        
        FILE *results_file    = fopen(results_file_name,    "w"); // Truncate file if it already exists. This ensures that we don't have partial results from being moved between cluster nodes when on all queue.
        if (results_file == 0) {cout << "Could not open results file. Exiting." << endl; printf("%s",results_file_name); exit(-1);}
        fclose (results_file);
        char spikes_file_name           [250]; sprintf(spikes_file_name,           "%sspike_times_%d.txt",           save_path, work_unit);
        char bursts_file_name           [250]; sprintf(bursts_file_name,           "%sburst_times_%d.txt",           save_path, work_unit); // Burst times defined by A current.
        char bursts_file_name_by_spikes [250]; sprintf(bursts_file_name_by_spikes, "%sburst_times_by_spikes_%d.txt", save_path, work_unit); // Burst times defined by timing of spikes (within period defined by A current).
        FILE *spikesfile           = fopen(spikes_file_name,           "w");
        FILE *burstsfile           = fopen(bursts_file_name,           "w");
        FILE *burstsfile_by_spikes = fopen(bursts_file_name_by_spikes, "w");
        fclose(burstsfile);
        fclose(burstsfile_by_spikes);
        fclose(spikesfile);
        
        
        for ( int sim_number=1; sim_number <= num_sims; ++sim_number){  
            cout << "Running sim number: " << sim_number << endl;
                            
            getQ10values(sim_number, num_sims);  // Get a set of Q10s.
            
            // Write out the Q10 values to the results file. (Actual results follow on the same line.)
            results_file = fopen(results_file_name,    "a");
            if (results_file == 0) {cout << "Could not open results file. Exiting." << endl; exit(-1);}
            fprintf(results_file,    "%f ", dt_hist);
            fprintf(results_file,    "%d ", num_temp_steps);
            for (int channel = 0; channel<num_channels; ++channel) { fprintf(results_file,    "%f %f ", Q10_alphas[0][channel],  Q10_alphas[1][channel]); } // Save activation, then inactivation.
            for (int channel = 0; channel<num_channels; ++channel) { fprintf(results_file,    "%f %f ", Q10_betas [0][channel],  Q10_betas [1][channel]); }
            for (int channel = 0; channel<num_channels; ++channel) { fprintf(results_file,    "%f ", Q10_g_bars[channel]);    }
            fprintf(results_file,    "%f %f %f %f %f ", Q10_Ca, Q10_axial[0], Q10_axial[1], Q10_gap, Q10_leak);
            fclose (results_file);
            
            // Do temperature ramp for single set of Q10s.
            for (int temp_step = 0; temp_step < num_temp_steps; ++temp_step){
                double temperature = min_temperature + temp_step_size*temp_step;
                cout << "temperature=" << temperature << endl;
                run_one_sim(temperature, sim_number, num_sims, work_unit);
            }
            
            results_file = fopen(results_file_name, "a");
            fprintf(results_file, "\n");
            fclose (results_file);
        }
    }
}

// Load Q10 values
void getQ10values(int sim_number, int num_sims){
    
    Q10_Ca       = diffusion_Q10;
    Q10_axial[0] = diffusion_Q10;
    Q10_axial[1] = diffusion_Q10;
    Q10_gap      = diffusion_Q10;
    Q10_leak     = diffusion_Q10;
    
    // Set alpha and beta Q10s.
    for (int activation=0; activation < 2; ++activation){ // 0 == activation; 1==inactivation.
        for (int channel = 0; channel<num_channels; ++channel){
            
            switch (Q10_selection_method){
                
                case constant:
                    diffusion_Q10 = identical_Q10s_value; // A test value to use for all Q10s. 
                    Q10_alphas[activation][channel] = identical_Q10s_value;
                    Q10_betas [activation][channel] = identical_Q10s_value;
                    Q10_Ca                          = identical_Q10s_value;
                    Q10_axial[0]                    = identical_Q10s_value;
                    Q10_axial[1]                    = identical_Q10s_value;
                    Q10_gap                         = identical_Q10s_value;
                    Q10_leak                        = identical_Q10s_value;
                    break;
                    
                case randomize:
                    Q10_alphas[activation][channel] = getQ10_alpha_beta_values(channel, activation, sim_number); 
                    if (use_same_alpha_and_beta){
                        Q10_betas [activation][channel] = Q10_alphas[activation][channel]; // Use alpha value for beta.
                    }else{
                        Q10_betas [activation][channel] = getQ10_alpha_beta_values( channel, activation, sim_number); // Select new random value for beta.
                    }
                    if (randomize_Q10_Ca){ // Select a random Q10 for Ca++ buffering, rather than the default diffusion Q10 value.
                        Q10_Ca       = random_doubles[sim_number-1][num_channels * 2 + 0]   * (Q10_max - Q10_min) + Q10_min; // Select next random value from pool and scale it.
                    }
                    if (randomize_Q10_axial){
                        Q10_axial[0] = random_doubles[sim_number-1][num_channels * 2 + 1 ]   * (Q10_max - Q10_min) + Q10_min; 
                        Q10_axial[1] = random_doubles[sim_number-1][num_channels * 2 + 2 ]   * (Q10_max - Q10_min) + Q10_min;
                    }
                    if (randomize_Q10_gap)  {
                        Q10_gap      = random_doubles[sim_number-1][num_channels * 2 + 3 ]   * (Q10_max - Q10_min) + Q10_min;
                    }
                    if (randomize_Q10_leak) {
                        Q10_leak     = random_doubles[sim_number-1][num_channels * 2 + 4 ]   * (Q10_max - Q10_min) + Q10_min;
                    }
                    break;
                    
                case slice:{
                    // To do a slice I need the slice position for the sliced dimensions and the ranges for the remaining.
                    // This could be simplified as a min and max for all dimensions which get evenly sampled.
                    // I also need to find my ID to know what value to use for the free dimensions.
                    int sim_ID = (job_ID/num_base_sets) * num_sims + sim_number -1; // NOTE: sim_number is indexed from 1, not zero. Division by num_base_sets ensures all members of base set get same Q10s

                    
                    
                    if ( ( (act_1 == activation && cond_1 == channel ) || (act_2 == activation && cond_2 == channel ) ) ) { // See if this Q10 needs to vary from default value.
                    // If this parameter has a range, calculate the range.
                        
                        //Calculate index values for free parameters.
                        int free_variable_index_1 = act_1 * num_channels + cond_1;
                        int index                 = activation * num_channels + channel; // Index of current Q10 variable
                        
                        bool this_index_is_first = false ; // Gotta know if this variable is first so we can index through the pair properly. If first, increment every time and wrap. If second, increment only when 1st rolls over.
                        if (free_variable_index_1 == index) { // Check if we are first.
                            this_index_is_first = true;
                        }
                        
                        // Now find the index of the free variable ...
                        int sample_point = -999;
                        if (this_index_is_first){
                            sample_point  = sim_ID % num_slice_sample_points;
                        }else{
                            sample_point  = sim_ID / num_slice_sample_points;
                        }
                        // ... and use those to get their values
                        double min = Q10_alpha_min[activation][channel];
                        double max = Q10_alpha_max[activation][channel];
                        Q10_alphas[activation][channel] = min + (max-min) / (num_slice_sample_points - 1) * sample_point; //  The -1 ensures that we really hit max, not just get near it. (Consider case of two sample points.)
                        
                        
                    }
                    cout << channel << " " << activation << " " << Q10_alphas[activation][channel] << endl;
                    
                    
                    if (use_same_alpha_and_beta){
                        Q10_betas [activation][channel] = Q10_alphas[activation][channel]; // Use alpha value for beta.
                    }else{
                        cout << "Must have same alpha and beta Q10 values when using slice code. Exiting." << endl; exit(-1);
                    }
                    break;
                }
                

                case single_points: // Test individual points in Q10 space.
                    Q10_alphas[activation][channel] = single_points_default_value;
                    Q10_betas [activation][channel] = single_points_default_value;
                    switch (sim_number){ // Indexed from 1.
                        case 1:
                            Q10_alphas[0][KCa] = 2;
                            Q10_betas[0][KCa]  = 2;                            
                            Q10_alphas[0][CaS] = 1;
                            Q10_betas[0][CaS]  = 1;                            
                            break;
                        case 2:
                            Q10_alphas[0][KCa] = 2;
                            Q10_betas[0][KCa]  = 2;                            
                            Q10_alphas[0][CaS] = 1.5;
                            Q10_betas[0][CaS]  = 1.5;                            
                            break;
                        case 3:
                            Q10_alphas[0][KCa] = 2;
                            Q10_betas[0][KCa]  = 2;                            
                            Q10_alphas[0][CaS] = 2.0;
                            Q10_betas[0][CaS]  = 2.0;                            
                            break;
                        case 4:
                            Q10_alphas[0][KCa] = 2;
                            Q10_betas[0][KCa]  = 2;                            
                            Q10_alphas[0][CaS] = 2.5;
                            Q10_betas[0][CaS]  = 2.5;                            
                            break;
                        case 5:
                            Q10_alphas[0][KCa] = 2;
                            Q10_betas[0][KCa]  = 2;                            
                            Q10_alphas[0][CaS] = 3.0;
                            Q10_betas[0][CaS]  = 3.0;                            
                            break;
                        case 6:
                            Q10_alphas[0][KCa] = 2;
                            Q10_betas[0][KCa]  = 2;                            
                            Q10_alphas[0][CaS] = 3.5;
                            Q10_betas[0][CaS]  = 3.5;                            
                            break;
                    }
                               
                    break;
                
               default:
                    cout << "Unknown Q10_selection_method. Exiting." << endl; exit(-1);
            }
        }
    }   
    

    // Set Q10 for maximal channel conductances.
    for (int channel = 0; channel<num_channels; ++channel){
        Q10_g_bars[channel] = diffusion_Q10; 
    }
    

    
    // Override Q10 values if we are using preset numbers from the literature.
    if (use_preset_Q10_for_select_channels){
        // Values form Frankenhaeuser and Moore 1963.
        Q10_alphas[0][Na]    = 1.84;
        Q10_alphas[1][Na]    = 2.80;
        Q10_betas [0][Na]    = 1.68;
        Q10_betas [1][Na]    = 2.93;
        Q10_alphas[0][K_AIS] = 3.20;
        Q10_betas [1][K_AIS] = 2.76;
        Q10_g_bars[Na]       = 1.3;
        Q10_g_bars[K_AIS]    = 1.2;
        Q10_g_bars[K_soma]   = 1.2;
    }
    
    // Ensure that channel dynamics are same for Kd, no matter where it is located.
    if (use_same_Q10s_for_both_Kds){ 
        Q10_alphas[0][K_soma] = Q10_alphas[0][K_AIS]; // Opening rate
        Q10_betas [0][K_soma] = Q10_betas [0][K_AIS]; // Closing rate
        Q10_g_bars[K_soma]    = Q10_g_bars[K_AIS];    // Maximal conductance
    }
    
    if (true ) {  // Display Q10_alphas and Q10_betas values.
        cout << "Q10_alphas: ";
        for (int k = 0; k<10; ++k){
            cout << Q10_alphas[0][k] << " ";
            cout << Q10_alphas[1][k] << " ";
        }
        cout << endl;
        cout << "Q10_betas: ";
        for (int k = 0; k<10; ++k){
            cout << Q10_betas[0][k] << " ";
            cout << Q10_betas[1][k] << " ";
        }
        cout << endl;
    }
}

// Return value between Q10_min and Q10_max with linear or ln spread.
double getQ10_alpha_beta_values(int channel_number, int activation, int sim_number){
    
    if (channel_number == K_soma){ // See if this is the soma's Kd channel
        return Q10_alphas[activation][1];  // If so, return the Q10 value of the axon's Kd channel, since these values should be the same.
    }else{
        double value   = 0; // Q10 value to be returned.
        value = random_doubles[sim_number-1][channel_number + activation*num_channels]  * (Q10_max - Q10_min) + Q10_min; 

        // Set Q10 value to 1 for inactivation of channels which don't have inactivation. 
        // (This has no effect on the simulation - it is just included for clarity.)
        if (activation == 1){ // Check whether the activation variable is on its second value - "inactivation".
            switch (channel_number){
                case 0: case 2: case 4: case 8: // Channels which inactivate
                    return value; 
                    break;
                case 1: case 3: case 5: case 6: case 7: case 9: // Channels which don't inactivate.
                    return 1; // Use Q10 of 1 to indicate that nothing happens here.
                    break;
                default:
                    cout << "Unknown channel number. Exiting." << endl;
                    exit (-1);
            }
        }else{        
            return value;
        }
    }
}   

int run_one_sim(double temperature, int sim_number, int num_sims, int work_unit){
    
    cout << "Running sim." << endl;

    double sim_length   = 30;  // [seconds]  Constant sim length used, since testing for return to prior state proved problematic.
    double dt= 0.005 * 0.001; // [seconds]  Initial time step size. This will change since this uses variable time step size. (Kept short to be conservative.)
    
    struct timeval  start_time, end_sim_time, end_time;
    gettimeofday(&start_time, NULL);

    int rng_seed =  sim_number + work_unit*num_sims; // Select unique seed to use for randomizing g_maxs.

    network_stats the_network_stats=model( temperature,  
               Q10_alphas,  
               Q10_betas,  
               Q10_g_bars,  
               Q10_Ca,  
               Q10_axial,  
               Q10_gap,  
               Q10_leak,  
               sim_length,  
               dt,
               dt_hist,
               cluster,
               randomize_initial_conductances,
               g_max_set,
               constant_Ca,
	       rng_seed
               );

    gettimeofday(&end_sim_time, NULL);

    printf("Mean burst length:     AB=%f   PD=%f\n", the_network_stats.AB_stats.mean_burst_length,     the_network_stats.PD_stats.mean_burst_length);
    printf("Duty cycle:            AB=%f   PD=%f\n", the_network_stats.AB_stats.duty_cycle,            the_network_stats.PD_stats.duty_cycle);
    printf("Duty cycle by spikes:  AB=%f   PD=%f\n", the_network_stats.AB_stats.duty_cycle_by_spikes,  the_network_stats.PD_stats.duty_cycle_by_spikes);
    printf("Spikes per burst:      AB=%f   PD=%f\n", the_network_stats.AB_stats.mean_spikes_per_burst, the_network_stats.PD_stats.mean_spikes_per_burst);
    printf("Mean burst frequency:  AB=%f   PD=%f\n", the_network_stats.AB_stats.mean_burst_frequency,  the_network_stats.PD_stats.mean_burst_frequency);
    
    
    // Add simulation results to the results file.
    FILE *results_file = fopen(results_file_name, "a");
    
    //fprintf(results_file, "%f %d ", temperature, the_network_stats.classification);
    fprintf(results_file, "%f %f ", the_network_stats.AB_stats.mean_burst_length,     the_network_stats.PD_stats.mean_burst_length);
    fprintf(results_file, "%f %f ", the_network_stats.AB_stats.duty_cycle,            the_network_stats.PD_stats.duty_cycle);
    fprintf(results_file, "%f %f ", the_network_stats.AB_stats.duty_cycle_by_spikes,  the_network_stats.PD_stats.duty_cycle_by_spikes);
    fprintf(results_file, "%f %f ", the_network_stats.AB_stats.mean_spikes_per_burst, the_network_stats.PD_stats.mean_spikes_per_burst);
    fprintf(results_file, "%f %f ", the_network_stats.AB_stats.mean_burst_frequency,  the_network_stats.PD_stats.mean_burst_frequency);
    fprintf(results_file, "%f %f ", the_network_stats.AB_stats.fraction_bursting,     the_network_stats.PD_stats.fraction_bursting);

    fprintf(results_file, "%f %f ", the_network_stats.AB_stats.V_min,                 the_network_stats.PD_stats.V_min);
    fprintf(results_file, "%f %f ", the_network_stats.AB_stats.V_max,                 the_network_stats.PD_stats.V_max);
    
    if (randomize_initial_conductances){
        for (int neuron = 0; neuron<num_neurons; ++neuron){
            for (int conductance = 0; conductance < num_channels; ++conductance ){
                fprintf(results_file, "%12.12f ", the_network_stats.g_max_all[neuron][conductance]); // Save the conductance BEFORE Q10 scaling. (Ensuring enough significant digits.)
            }
        }
    }
    
    fclose(results_file);
    
    if (save_spikes_and_bursts){
        // Save spike and burst timing data.
        char spikes_file_name           [250]; sprintf(spikes_file_name,           "%sspike_times_%d.txt",           save_path, work_unit);
        char bursts_file_name           [250]; sprintf(bursts_file_name,           "%sburst_times_%d.txt",           save_path, work_unit); // Burst times defined by A current.
        char bursts_file_name_by_spikes [250]; sprintf(bursts_file_name_by_spikes, "%sburst_times_by_spikes_%d.txt", save_path, work_unit); // Burst times defined by timing of spikes (within period defined by A current).
        FILE *spikesfile           = fopen(spikes_file_name,           "a");
        FILE *burstsfile           = fopen(bursts_file_name,           "a");
        FILE *burstsfile_by_spikes = fopen(bursts_file_name_by_spikes, "a");
        // Save sim_number and temperature
        fprintf(burstsfile,           "%d %f \n", sim_number, temperature);
        fprintf(burstsfile_by_spikes, "%d %f \n", sim_number, temperature);
        // Save all burst onset and off set times
        for (unsigned int i=0; i< the_network_stats.AB_burst_onsets.size();  ++i){ fprintf(burstsfile, "%f ", the_network_stats.AB_burst_onsets[i]); } fprintf(burstsfile, "\n");
        for (unsigned int i=0; i< the_network_stats.AB_burst_offsets.size(); ++i){ fprintf(burstsfile, "%f ", the_network_stats.AB_burst_offsets[i]);} fprintf(burstsfile, "\n");
        for (unsigned int i=0; i< the_network_stats.PD_burst_onsets.size() ; ++i){ fprintf(burstsfile, "%f ", the_network_stats.PD_burst_onsets[i]); } fprintf(burstsfile, "\n");
        for (unsigned int i=0; i< the_network_stats.PD_burst_offsets.size(); ++i){ fprintf(burstsfile, "%f ", the_network_stats.PD_burst_offsets[i]);} fprintf(burstsfile, "\n");
        
        for (unsigned int i=0; i< the_network_stats.AB_bursts_on.size();  ++i){ fprintf(burstsfile_by_spikes, "%f ", the_network_stats.AB_bursts_on[i]); } fprintf(burstsfile_by_spikes, "\n");
        for (unsigned int i=0; i< the_network_stats.AB_bursts_off.size(); ++i){ fprintf(burstsfile_by_spikes, "%f ", the_network_stats.AB_bursts_off[i]);} fprintf(burstsfile_by_spikes, "\n");
        for (unsigned int i=0; i< the_network_stats.PD_bursts_on.size() ; ++i){ fprintf(burstsfile_by_spikes, "%f ", the_network_stats.PD_bursts_on[i]); } fprintf(burstsfile_by_spikes, "\n");
        for (unsigned int i=0; i< the_network_stats.PD_bursts_off.size(); ++i){ fprintf(burstsfile_by_spikes, "%f ", the_network_stats.PD_bursts_off[i]);} fprintf(burstsfile_by_spikes, "\n");
        // Save all spike onset times.
        fprintf(spikesfile, "%d %f \n", sim_number, temperature);
        for (unsigned int i=0; i< the_network_stats.AB_spike_onsets.size();  ++i){ fprintf(spikesfile, "%f ", the_network_stats.AB_spike_onsets[i]); } fprintf(spikesfile, "\n");
        for (unsigned int i=0; i< the_network_stats.PD_spike_onsets.size();  ++i){ fprintf(spikesfile, "%f ", the_network_stats.PD_spike_onsets[i]); } fprintf(spikesfile, "\n");
        
        fclose(burstsfile);
        fclose(burstsfile_by_spikes);
        fclose(spikesfile);
    }
    
    // Save lots of history information if running on desktop.
    if (!cluster){
        char voltage_file_name        [250]; sprintf(voltage_file_name,         "%svoltage_%d_%d.txt",         save_path, sim_number, (int)temperature);
        char current_file_name        [250]; sprintf(current_file_name,         "%scurrent_%d_%d.txt",         save_path, sim_number, (int)temperature);
        char activation_file_name     [250]; sprintf(activation_file_name,      "%sactivation_%d_%d.txt",      save_path, sim_number, (int)temperature);
        char inactivation_file_name   [250]; sprintf(inactivation_file_name,    "%sinactivation_%d_%d.txt",    save_path, sim_number, (int)temperature);
        char calcium_file_name        [250]; sprintf(calcium_file_name,         "%scalcium_%d_%d.txt",         save_path, sim_number, (int)temperature);
        char return_to_state_file_name[250]; sprintf(return_to_state_file_name, "%sreturn_to_state_%d_%d.txt", save_path, sim_number, (int)temperature);
        char total_dist_file_name     [250]; sprintf(total_dist_file_name,      "%stotal_dist_%d_%d.txt",      save_path, sim_number, (int)temperature);
        char edge_dist_file_name      [250]; sprintf(edge_dist_file_name,       "%sedge_dist_%d_%d.txt",       save_path, sim_number, (int)temperature);
        
        // Save Voltage
        if (save_voltage){
            FILE *voltagefile      = fopen(voltage_file_name,     "w");
            for (int i=1; i<the_network_stats.save_slots; ++i){
                fprintf(voltagefile, "%1.20f,%1.20f, %1.20f,%1.20f,\n", the_network_stats.V_hist[AB][Soma][i], the_network_stats.V_hist[AB][AIS][i],  the_network_stats.V_hist[PD][Soma][i], the_network_stats.V_hist[PD][AIS][i]); // [neuron][compartment][time_step] :.  AB_Soma, AB_AIS, PD_Soma, PD_AIS.
            }
            fclose(voltagefile);
        }

        // Save [Ca++]
        if (save_Ca){
            FILE *calciumfile = fopen(calcium_file_name,     "w");
            for (int i=1; i<the_network_stats.save_slots; ++i){
                fprintf(calciumfile, "%1.20f,%1.20f,\n", the_network_stats.Ca_hist[AB][i], the_network_stats.Ca_hist[PD][i]); // [neuron][time_step]
            }
            fclose(calciumfile);
        }
        
        // Save channel currents
        if (save_current){
            FILE *currentfile      = fopen(current_file_name,     "w");
            for (int i=1; i<the_network_stats.save_slots; ++i){
                for (int j=0; j<num_channels; ++j){
                    fprintf(currentfile,      "%1.20f,%1.20f,", the_network_stats.I_hist[AB][j][i], the_network_stats.I_hist[PD][j][i]); //[neuron][channel][time_step]
                }
                fprintf(currentfile,      "\n");
            }
            fclose(currentfile);
        }
        
        // Save ALL channel state variables
        if (save_state_vars){
            FILE *activationfile   = fopen(activation_file_name,  "w");
            FILE *inactivationfile = fopen(inactivation_file_name,"w");
            for (int i=1; i<the_network_stats.save_slots; ++i){
                for (int j=0; j<num_channels; ++j){
                    fprintf(activationfile,   "%1.20f,%1.20f,", the_network_stats.m_hist[AB][j][i], the_network_stats.m_hist[PD][j][i]); //[neuron][channel][time_step]
                }
                fprintf(activationfile,   "\n");
            }
            fclose(activationfile);
        
            for (int i=1; i<the_network_stats.save_slots; ++i){
                for (int j=0; j<num_channels; ++j){
                    fprintf(inactivationfile, "%1.20f,%1.20f,", the_network_stats.h_hist[AB][j][i], the_network_stats.h_hist[PD][j][i]); //[neuron][channel][time_step]
                }
                fprintf(inactivationfile, "\n");
            }
            fclose(inactivationfile);
        }
        
    }  
        
    gettimeofday(&end_time, NULL);
    
    printf ("Simulation time: %f\n",   difftime_ms(end_sim_time , start_time  )/1000.0);
    if (!cluster) printf ("Save time:       %f\n", difftime_ms(end_time     , end_sim_time)/1000.0);
    if (!cluster) printf ("Total time:      %f\n", difftime_ms(end_time     , start_time  )/1000.0);
    printf("\n");
       
    return(0);
}

int difftime_ms(timeval t1, timeval t2){ // Return the difference in two time values.
    return (int)(((t1.tv_sec - t2.tv_sec) * 1000000) +  (t1.tv_usec - t2.tv_usec))/1000;
}

void load_rand_numbers(int num_sims){
    
    // Set the seed to 1. This ensures that the random number stream is _repeatable_. 
    unsigned seed = 1; // NOTE: We are using a constant value for the seed rather than something like: std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator (seed);  // mt19937 is a standard mersenne_twister_engine
    
    int numbers_per_sim  = num_channels * 2; // How many random numbers are required per sim.
    if (Q10_selection_method == randomize && (randomize_Q10_Ca || randomize_Q10_axial || randomize_Q10_gap || randomize_Q10_leak )){
        numbers_per_sim += 5; 
    }
    
    int sim_groups_to_skip = job_ID / num_base_sets; // A group of sims is a set of jobs that use the same Q10s but with different base conductance sets.
    
    // Advance the random number generator stream past all the numbers used by prior jobs.
    for (int sim_group = 0; sim_group < sim_groups_to_skip; ++sim_group){
        for (int j =0; j < num_sims; j++){ // num_sims is how many simulations each job does.
            for (int i =0; i< numbers_per_sim; i++){ 
                //cout << "Skipping ... Group " << sim_group << "  sim_number " << j  << " Random value = " << (double)generator()/(double)(generator.max()) << endl ; // Call the RNG, discarding the output. Generates compiler warning.
                (double)generator(); // Call the RNG, discarding the output. Generates compiler warning on unused value.
            }
        }
    }

    // Read all randoms needed by this job into an array.
    for (int i=0; i<num_sims; ++i){
        for (int j=0; j<numbers_per_sim; ++j){
            random_doubles[i][j] = (double)generator()/(double)(generator.max());
        }
    }
}


void setup_Slice_code(){
    
    // Set all Q10 values for channels to the default value.
    for(int i=0;i<2;i++){
        for(int j=0;j<num_channels;j++){
            Q10_alpha_min [i][j] = default_slice_Q10_value;
            Q10_alpha_max [i][j] = default_slice_Q10_value;
            Q10_alphas    [i][j] = default_slice_Q10_value;
            Q10_betas     [i][j] = default_slice_Q10_value;
        }
    }
    
    // Set values which are not default. Should be passed from command line.
    Q10_alpha_min[act_1][cond_1] = varied_Q10_min;
    Q10_alpha_max[act_1][cond_1] = varied_Q10_max;
    
    Q10_alpha_min[act_2][cond_2] = varied_Q10_min;
    Q10_alpha_max[act_2][cond_2] = varied_Q10_max;
    

}
