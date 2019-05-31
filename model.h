#ifndef MODEL_H_
#define MODEL_H_
#endif /* MODEL_H_ */


//#define testing // Use this #define to enable some sanity checks. Turn off for speed.
#define desktop // Enables recording of state information like voltage, channel state and [Ca++].


using namespace std;
#include <vector>
#include <sys/time.h>

const int num_channels = 10;
const int num_neurons  =  2;

enum currents     { Na=0, K_AIS=1, CaT=2, CaS=3, nap=4, H=5, K_soma=6, KCa=7, A=8, proc=9, Leak_AIS=10, Leak_soma=11, Axial=12, Gap=13};
enum cellTypes    { AB   = 0, PD  = 1};
enum compartments { Soma = 0, AIS = 1}; // (AIS: Axon initial segment.)

struct neuron_stats {
    double mean_spikes_per_burst;
    double mean_burst_length;
    double mean_burst_frequency;
    double duty_cycle;
    double duty_cycle_by_spikes;
    double fraction_bursting; // Fraction of the time that a cell is bursting. Useful for irregular bursters where duty cycle can't be accurately calculated.
    bool   regular_bursting;  // true if the CV of inter-burst interval is <0.05 and there are at least 3 bursts. 
    double V_min;             // Minimum voltage reached after stabilization period.
    double V_max;             // Maximum  ... 
};

struct network_stats{
    
    neuron_stats AB_stats;
    neuron_stats PD_stats;
    int save_slots;

    
    vector<vector<vector<double> > > V_hist;

    #ifdef desktop
        vector<vector<vector<double> > > I_hist;
        vector<vector<vector<double> > > m_hist;
        vector<vector<vector<double> > > h_hist;
        vector<vector<double> >          Ca_hist;
    #endif
        
    vector <double> AB_burst_onsets;  // Classified by A current activation.
    vector <double> PD_burst_onsets;
    vector <double> AB_burst_offsets;
    vector <double> PD_burst_offsets;
    
    vector <double> AB_spike_onsets;
    vector <double> PD_spike_onsets;
    
    vector <double> AB_bursts_on;    // Classified by spike times within burst. (Note difference from burst_onsets.)
    vector <double> AB_bursts_off;
    vector <double> PD_bursts_on;
    vector <double> PD_bursts_off;

    vector <vector<double> > g_max_all_adj; // Maximal conductance     adjusted for Q10 values.
    vector <vector<double> > g_max_all;     // Maximal conductance NOT adjusted for Q10 values.
    vector <double>          g_axial;
    double                   g_gap;
    vector <vector<double> > g_leaks;
};

// Function Prototypes.

network_stats    model(
    double delta_temperature, 
    double Q10_alphas[2][10], 
    double Q10_betas[2][10], 
    double Q10_g_bars[10],
    double Q10_Ca,
    double Q10_axial[2],
    double Q10_gap, 
    double Q10_leak,
    double sim_length,
    double dt,
    double dt_hist,  /* Sampling interval for saving state. */
    bool   cluster,
    bool   randomize_initial_conductances,
    int    g_max_set,
    bool   constant_Ca,
    int    rng_seed
    );
    //, int save_slots);

void    oneStep(int last_step, int next_step, double dt);

double  sigmoid(double a, double b, double c, double  V);

double  tau_sigmoid(double a, double b, double c, double d, double e, double V);

int     get_channel_state(int neuron, int channel, double V, double Calcium, double old_m, double old_h, double dt, double Q_m_alpha, double Q_h_alpha, double Q_m_beta, double Q_h_beta, double * act, double * inact);

double  my_mean (vector <double> input, double default_for_div0);
double  standard_deviation(vector <double> input);


bool    AlmostEqual2sComplement(float A, float B, int maxUlps);

int     difftime_ms(timeval t1, timeval t2);
