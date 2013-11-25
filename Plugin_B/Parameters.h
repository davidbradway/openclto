// TODO: these should be moved to into Int Parameter Array
#define DATA_SIZE_IN (1136*75*32)
#define DATA_SIZE_OUT (1136*75)
//#define DATA_SIZE_IN (1136*750*32)
//#define DATA_SIZE_OUT (1136*750)

/// <summary> Integer Parameter Array Enumerated Indices </summary>
typedef enum IntParamIndex {
	ind_emissions,     //Number of emissions in same direction (ensemble length)
	ind_nlines,        //Number of places and/or number of ensembles used
	ind_nlinesamples,  //Number of samples per line
	ind_numb_avg,      //Number of estimates to average over
	ind_avg_offset,    //
	ind_lag_axial,     // = 1, //
	ind_lag_TO,        // = 2, //
	ind_lag_acq,       // = 1, //
	IntParamCount
};

/// <summary> Float Parameter Array Enumerated Indices </summary>
typedef enum FloatParamIndex {
	ind_fs,       //The sampling freqency. [Hz]
	ind_f0,       //The central frequency of the excitation. [Hz]
	ind_c,        //The speed of sound. [m/s]
	ind_fprf,     //Pulse repetition frequency [Hz]
	ind_depth,    // = 0.03, // 3cm focal depth
	ind_lambda_X, //.0033 m
	FloatParamCount
};

// TODO: the ParamStruct should be removed.
/// <summary> Structure containing scanner parameters </summary>
typedef struct ParamStruct{
	int emissions; //Number of emissions in same direction
	int nlines;
	int nlinesamples;
	int numb_avg; //Number of estimates to average over
	int avg_offset;
	int lag_axial; // = 1; //
	int lag_TO; // = 2; //
	int lag_acq; // = 1; //bradway

	float fs; //The sampling freqency. [Hz]
	float f0; //The central frequency of the excitation. [Hz]
	float c; //The speed of sound. [m/s]
	float fprf; //Pulse repetition frequency [Hz]
	float depth; // = 0.03; // 3cm focal depth
	float lambda_X; //.0033 m
} ParamStruct;
