/*	Kernel file for use in plugin library
 *	Made for velocity estimations by autocorrelation
 *	axial and transverse dimensions
 */

/** Kernel for splitting inbuf into intermediate buffers
 *	@param inbuf - OpenCL buffer containing packed data
 *	@param nlinesamples - integer Number of samples in each line (i.e. 1136)
 *	@param inbufZ - OpenCL buffer containing packed real part of axial data
 *	@param inbufLR - OpenCL buffer containing packed real part of Left beam
 *	@param N - number of lines*emissions, i.e. memory blocks to split
 */
__kernel void split(__global short2* inbuf,
					  const  int     nlinesamples,
					  const  int     nlines,
					  const  int     interleave,
					  const  int     emissions,
					__global float2* Z,
					__global float2* Z2,
					__global float2* L,
					__global float2* R) {
 	// unwrap single inbuf into separate buffers
	// did assume inbuf will have 'dimensions' in this order [real/imag, line samples, position (Z1/Z2/L/R), lines, emission shots]
	// now assume inbuf will have 'dimensions' in this order [real/imag, nlinesamples, interleave =Z1/Z2/L/R * position, emissions shots, =nlines/interleave]
	size_t local_size = get_local_size(0); // how big is this group? 64
	size_t group_id =  get_group_id(0); // which group number is this?
	size_t local_id = get_local_id(0); // where am i in this group of 64
	size_t global_size = get_global_size(0);  // roundup(1136*4*75*32,64)
	size_t global_id = get_global_id(0); // i.e. 66
	size_t i,j,k;
	int latgroups = nlines/(interleave/4);
	/*
	// Ask Lee about this:
	// http://stackoverflow.com/questions/15394882/need-help-understanding-opencl-reductions
	// consider contiguous memory access
	// need to understand while vs if
	// in parallel reduction, how many threads are used?
	//
	// OR just index into the inbuf in the right place later in the program
	// and figure out integer math rather than converting to floats?
	//
	*/
	// make a kernel that has a work item for each of nlinesamples
	// global_id is the sample in depth
	if (global_id < nlinesamples) {
		// in each thread, loop across interleave=Z/Z2/L/R*locations, emissions, latgroups=nlines/interleave
		// reorder so emissions is in last dimension
		// want result in this order: locations, =nlines/interleave, emissions, 
		for(k=0;k<latgroups;k++){ //lateral group counter: 0-24 or 0-6
			for(j=0;j<emissions;j++){ //emission counter: 0-15 or 0-31
				for(i=0;i<interleave;i++){ //interleave counter: 0-15 or 0-11
					if     (i%4==0){       Z[j*latgroups*(interleave/4)*nlinesamples + k*(interleave/4)*nlinesamples + (i/4)*nlinesamples + global_id] = 
						convert_float2(inbuf[k*emissions* interleave   *nlinesamples + j* interleave   *nlinesamples +  i   *nlinesamples + global_id]);
					}
					else if(i%4==1){      Z2[j*latgroups*(interleave/4)*nlinesamples + k*(interleave/4)*nlinesamples + (i/4)*nlinesamples + global_id] = 
						convert_float2(inbuf[k*emissions* interleave   *nlinesamples + j* interleave   *nlinesamples +  i   *nlinesamples + global_id]);
					}
					else if(i%4==2){	   L[j*latgroups*(interleave/4)*nlinesamples + k*(interleave/4)*nlinesamples + (i/4)*nlinesamples + global_id] = 
						convert_float2(inbuf[k*emissions* interleave   *nlinesamples + j* interleave   *nlinesamples +  i   *nlinesamples + global_id]); 
					}
					else           {	   R[j*latgroups*(interleave/4)*nlinesamples + k*(interleave/4)*nlinesamples + (i/4)*nlinesamples + global_id] = 
						convert_float2(inbuf[k*emissions* interleave   *nlinesamples + j* interleave   *nlinesamples +  i   *nlinesamples + global_id]);
					}
				}
			}
		}
	}
	/*
	if (global_id < N) {
		// in each thread, loop down samples.
		// access the memory locations for IQ and Z,Z2,L,R
		for(i=0;i<nlinesamples;i++){
			Z[offset+i]  = convert_float2(inbuf[offset*(4) + i]);
		}
		//for(i=0;i<nlinesamples;i++){
		//	Z2[offset+i] = convert_float2(inbuf[offset*(4) + (nlinesamples) + i]);
		//}
		for(i=0;i<nlinesamples;i++){
			L[offset+i]  = convert_float2(inbuf[offset*(4) + (2*nlinesamples) + i]);
		}
		for(i=0;i<nlinesamples;i++){
			R[offset+i]  = convert_float2(inbuf[offset*(4) + (3*nlinesamples) + i]);
		}
	}
	*/
}

/** Kernel for calculating standard deviation of input array
 *	The standard deviation is calculated through mean and sumproduct
 *	@param data_re OpenCL buffer containing real data for standard deviation
 *	@param data_im OpenCL buffer containing imaginary data for standard deviation
 *	@param global_sum1_real OpenCL buffer to store summation for mean
 *	@param global_sum1_imag OpenCL buffer to store summation for mean
 *	@param global_sum2 OpenCL buffer to store sum product
 *	@param N Nsamples - Number of samples in 2D, meaning data(:,:,i)
 *	@param emissions Emissions in same direction
 *	@param result Final result - standard deviation of Nsamples
*/
__kernel void std_dev(__global float8* data, 
					  __global float*  global_sum1_real,
					  __global float*  global_sum1_imag,
					  __global float*  global_sum2, 
					    const  int     N, 
					    const  int     emissions, 
					  __global float*  result){
 	// numbers with 1 handles mean, 2 handles std deviation product sum
	float sum1_real, sum1_imag;
	float sum2;
	float4 sum_vector_real, sum_vector_imag;
	size_t global_addr = get_global_id(0)*2;
	size_t local_id = get_local_id(0);
	__local float local_sum1_real[64];
	__local float local_sum1_imag[64];
	__local float local_sum2[64];

	// sum for mean calculation
	float8 tmpdata  = data[global_addr];
	float8 tmpdata1 = data[global_addr+1];
	sum_vector_real = tmpdata.even + tmpdata1.even;
	sum_vector_imag = tmpdata.odd  + tmpdata1.odd;

	local_sum1_real[local_id] = sum_vector_real.s0 + sum_vector_real.s1 + sum_vector_real.s2 + sum_vector_real.s3;
	local_sum1_imag[local_id] = sum_vector_imag.s0 + sum_vector_imag.s1 + sum_vector_imag.s2 + sum_vector_imag.s3;
	
	// sum products for std dev calculation
	local_sum2[local_id] = dot(sum_vector_real ,sum_vector_real) + dot(sum_vector_imag,sum_vector_imag);
	
	// Wait until all local parallel threads in this group are done
	barrier(CLK_LOCAL_MEM_FENCE);

	// The first thread in the group sums the real's, imag's, and square's
	if(local_id == 0){
		sum1_real = 0.0f;
		sum1_imag = 0.0f;
	
		sum2 = 0.0f;
		for(int i=0; i<get_local_size(0); i++){
			sum1_real += local_sum1_real[i];
			sum1_imag += local_sum1_imag[i];
	
			sum2 += local_sum2[i];
		}
		// Write to global memory so other threads can read the sums.
		global_sum1_real[get_group_id(0)] = sum1_real;
		global_sum1_imag[get_group_id(0)] = sum1_imag;

		global_sum2[get_group_id(0)] = sum2;
	}
	// Wait until all groups are done
	// Need a second kernel to do this last bit
	// can't do a barrier that will wait for all the parallel threads to finish
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	if(global_addr == 0){
		sum1_real = 0.0f; sum1_imag = 0.0f;
		sum2 = 0.0f;
		for(int i=0; i<get_num_groups(0); i++){
			sum1_real += global_sum1_real[i];
			sum1_imag += global_sum1_imag[i];

			sum2 += global_sum2[i];
		}
		sum1_real /= N;
		sum1_imag /= N;
		
		//printf("%f\n", sqrt((sum2 - N*(sum1_real*sum1_real+sum1_imag*sum1_imag))/(N-1)));
		*result = sqrt((sum2 - N*(sum1_real*sum1_real+sum1_imag*sum1_imag))/(N-1));
	}
}

/**	Autocorrelation kernel for velocity estimation
 *	Takes a set of data from ultrasound scanner.
 *	Each work item calculates standard deviation and 
 *	autocorrelation at certain "depth". Results are to
 *	be used in arctan kernel for a velocity estimate.
 *	@param data_re OpenCL buffer containing real data for velocity estimation
 *	@param data_im OpenCL buffer containing imaginary data for velocity estimation
 *	@param global_temp_re OUTPUT OpenCL buffer containing real data from autocorrelation
 *	@param global_temp_im OUTPUT OpenCL buffer containing imaginary data from autocorrelation
 *	@param emissions Number of emissions in same direction
 *	@param Nsamples Number of samples in 2D, meaning data(:,:,i)
 *	@param std_dev_global INPUT Standard deviation in first Nsamples, calculated by std_dev kernel
 */
__kernel void velocity_est( __global float2* data,
							__global float* global_temp_re,
							__global float* global_temp_im,
							  const  int    emissions,
							  const  int    Nsamples,
							__global float* std_dev_global){
	size_t local_size = get_local_size(0), group_id = get_group_id(0),
		local_id = get_local_id(0), global_id = get_global_id(0);
	
	float sum_re = 0, sum_im = 0; 
	float avg_re, avg_im;
	float array_re[2], array_im[2];

	size_t i;

	// std dev calc
	float sum2 = 0.0f, std_dev;
	for(i=0;i<emissions;i++){
		float2 tmpdata  = data[global_id+Nsamples*i];

		sum_re += tmpdata.x;
		sum_im += tmpdata.y;
		
		// std dev calc
		//sum2 += data_re[global_id+Nsamples*i]*data_re[global_id+Nsamples*i]+data_im[global_id+Nsamples*i]*data_im[global_id+Nsamples*i];
	}
	avg_re = sum_re/emissions;
	avg_im = sum_im/emissions;

	// std dev calc
	//std_dev = sqrt((sum2 - emissions*(avg_re*avg_re+avg_im*avg_im))/(emissions-1)); // std dev calc

	sum_re = 0.0f;
	sum_im = 0.0f;
	//sum.x = 0.0f;
	//sum.y = 0.0f;

	// Subtract the mean (through the emission dimension) from the data
	for(i=0;i<emissions-1;i++){
		float2 tmpdata  = data[global_id+Nsamples*i];
		// FOR NOW, OMIT ECHO CANCELING
		array_re[0] = tmpdata.x; //- avg_re;
		array_im[0] = tmpdata.y; //- avg_im;
	
		float2 tmpdata1  = data[global_id+Nsamples*(i+1)];
		// FOR NOW, OMIT ECHO CANCELING
		array_re[1] = tmpdata1.x; //- avg_re;
		array_im[1] = tmpdata1.y; //- avg_im;
	
		// autocorrelation sum
		sum_re += array_re[0] * array_re[1] - (-array_im[0]) * array_im[1];
		sum_im += array_re[0] * array_im[1] + (-array_im[0]) * array_re[1];
	}
	global_temp_re[global_id] = sum_re;
	global_temp_im[global_id] = sum_im;
	
	// std dev calc, if low std dev through emission dimension then zero out autocorrelation data
	// Maybe the "deciding factor" (here: 10) should be user controlled?
	//if((int)std_dev < (int)(*std_dev_global/10)){
		// don't do this now. //bradway
		//global_temp_re[global_id] = 0.0f;
		//global_temp_im[global_id] = 0.0f;
	//}
}

/**	Kernel for calculating average and arctan2 of input arrays
 *	Handles the output from velocity_est kernel and
 *	returns the final velocity estimates
 *	@param data_re INPUT OpenCL buffer containing real data from autocorrelation
 *	@param data_im INPUT OpenCL buffer containing imaginary data from autocorrelation
 *	@param scale Scaling factor for after arctan2
 *	@param numb_avg Number of depths to average over
 *	@param avg_offset Step between each average
 *	@param global_result OUTPUT OpenCL buffer containing final velocity estimates
 */
__kernel void arctan(__global float* global_temp_re,
					 __global float* global_temp_im,
					   const  float  scale,
					   const  int    numb_avg,
					   const  int    avg_offset,
					 __global float* global_result){
  	size_t global_id = get_global_id(0),i;
	float sum_re = 0.0f,sum_im = 0.0f;

	// Note: this averages across the end of a line to the next one. or even out of bounds.
	// consider min(num_avg, nlinesamples - (global_id % linesamples)
	//for(i=0;i<(min(numb_avg,nlinesamples-(global_id%nlinesamples));i++){ // number to average over. 40=8/35*175
	for(i=0;i<numb_avg;i++){ //number to average over. 40=8/35*175
		sum_re += global_temp_re[global_id*avg_offset+i];
		sum_im += global_temp_im[global_id*avg_offset+i];
	}
	sum_re /= numb_avg;
	sum_im /= numb_avg;
	global_result[global_id]=-scale*atan2(sum_im,sum_re);
}

/**	to_velocity_est kernel for velocity estimation
 *	Takes a set of data from ultrasound scanner.
 *	Each work item calculates 
 *	autocorrelation at certain "depth". Results are to
 *	be used in arctan kernel for a velocity estimate.
 *	@param dataL_re INPUT OpenCL buffer containing real data for velocity estimation
 *	@param dataL_im INPUT OpenCL buffer containing imaginary data for velocity estimation
 *	@param dataR_re INPUT OpenCL buffer containing real data for velocity estimation
 *	@param dataR_im INPUT OpenCL buffer containing imaginary data for velocity estimation
 *	@param lag_TO Transverse lag
 *	@param emissions Number of emissions in same direction
 *	@param Nsamples Number of samples in 2D, meaning data(:,:,i)
 *	@param global_sum12_re_im OUTPUT OpenCL buffer containing data from autocorrelations
 */
__kernel void to_velocity_est(__global float2* dataL,
							  __global float2* dataR,
							    const  int     lag_TO,
							    const  int     emissions,
							    const  int     Nsamples,
							  __global float4* global_sum12_re_im){
  	size_t local_size = get_local_size(0), group_id = get_group_id(0),
		local_id = get_local_id(0), global_id = get_global_id(0);
	float2 avgL = 0, avgR = 0;
	float2 r_sq, r_sqh;
	float2 r1, r2;
	float2 r1_TO, r2_TO;
	float2 sum1, sum2;
	size_t i;
	float4 sum12_re_im = 0;

	// find the sum and average for each datapoint through emissions
	for(i=0;i<emissions;i++){
		float2 tmpL = dataL[global_id+Nsamples*i];
		float2 tmpR = dataR[global_id+Nsamples*i];
		avgL += tmpL;
		avgR += tmpR;
	}
	avgL /= emissions;
	avgR /= emissions;

	// Subtract the mean (through the emission dimension) from the data
	// and form the in-phase sampled and hilbert quadrature samples from the left and right beams
	for(i=0;i<emissions-lag_TO;i++){
		float2 tmpL = dataL[global_id+Nsamples*i];
		float2 tmpR = dataR[global_id+Nsamples*i];

		// FOR NOW OMIT ECHO CANCELING
		r_sq.x  = tmpL.x;// - avgL.x;
		r_sqh.x = tmpL.y;// - avgL.y;
		r_sq.y  = tmpR.x;// - avgR.x;
		r_sqh.y = tmpR.y;// - avgR.y;

		//%Create r1 and r2 according to [1]
		//r1 = r_sq + j*r_sqh;
		//r1 = (r_sq_re + j*r_sq_im) + j*(r_sqh_re + j*r_sqh_im);
		//r1 = r_sq_re + j*r_sq_im + j*r_sqh_re - r_sqh_im;
		r1.x = r_sq.x - r_sqh.y;
		r1.y = r_sq.y + r_sqh.x;
		
		//r2 = r_sq - j*r_sqh;
		//r2 = (r_sq_re + j*r_sq_im) - j*(r_sqh_re + j*r_sqh_im);
		//r2 = r_sq_re + j*r_sq_im - j*r_sqh_re + r_sqh_im;
		r2.x = r_sq.x + r_sqh.y;
		r2.y = r_sq.y - r_sqh.x;
		
		//reuse these local vars for storage of 'i+lag_TO' sample
		tmpL = dataL[global_id+Nsamples*(i+lag_TO)];
		tmpR = dataR[global_id+Nsamples*(i+lag_TO)];
		
		// FOR NOW OMIT ECHO CANCELING
		r_sq.x  = tmpL.x;// - avgL.x;
		r_sqh.x = tmpL.y;// - avgL.y;
		r_sq.y  = tmpR.x;// - avgR.x;
		r_sqh.y = tmpR.y;// - avgR.y;

		r1_TO.x = r_sq.x - r_sqh.y;
		r1_TO.y = r_sq.y + r_sqh.x;
		r2_TO.x = r_sq.x + r_sqh.y;
		r2_TO.y = r_sq.y - r_sqh.x;

		// This is the autocorrelation sum
		// sumX=sum(conj(rX(:,1:end-k1)).*rX(:,1+k1:end),2);
		// sum1+=(conj(r1(:,1:end-k1)).*r1(:,1+k1:end)
		// sum1+=(r1_re[i] - j*r1_im[i])*(r1_re[1+lag_TO] + j*r1_im[1+lag_TO])
		// sum1+=r1_re[i]*r1_re[1+lag_TO] + r1_re[i]*j*r1_im[1+lag_TO] - j*r1_im[i]*r1_re[1+lag_TO] - j*r1_im[i]*j*r1_im[1+lag_TO]
		// sum1+=(r1_re[i] * r1_re[1+lag_TO] - (-r1_im[i])*r1_im[1+lag_TO]) + j*(r1_re[i] * r1_im[1+lag_TO] + (-r1_im[i]) * r1_re[1+lag_TO])
		// sum1_re += r1_re[i] * r1_re[1+lag_TO] - (-r1_im[i]) * r1_im[1+lag_TO]
		// sum1_im += r1_re[i] * r1_im[1+lag_TO] + (-r1_im[i]) * r1_re[1+lag_TO]

		// Pack the 4 components into a float4
		sum12_re_im.x += r1.x * r1_TO.x - (-r1.y) * r1_TO.y; // sum1_re
		sum12_re_im.y += r1.x * r1_TO.y + (-r1.y) * r1_TO.x; // sum1_im
		sum12_re_im.z += r2.x * r2_TO.x - (-r2.y) * r2_TO.y; // sum2_re
		sum12_re_im.w += r2.x * r2_TO.y + (-r2.y) * r2_TO.x; // sum2_im
	}
	global_sum12_re_im[global_id] = sum12_re_im;
}

/**	to_arctanX kernel for calculating average and arctan2 of input arrays
 *	Handles the output from velocity_est kernel and
 *	returns the final velocity estimates
 *	@param global_sum12_re_imX INPUT OpenCL buffer containing sums from autocorrelations
 *	@param k_axial Scaling factor for after arctan2
 *	@param k_trans Scaling factor for after arctan2
 *	@param numb_avg Number of depths to average over
 *	@param avg_offset Step between each average
 *	@param nlinesamples number of axial samples per line
 *	@param global_axial_result      OUTPUT OpenCL buffer containing final velocity estimates
 *	@param global_transverse_result OUTPUT OpenCL buffer containing final velocity estimates
 */
__kernel void to_arctan(__global float4* global_sum12_re_im,
						  const  float   k_axial,
						  const  float   k_trans,
						  const  int     numb_avg,
						  const  int     avg_offset,
						  const  int     nlinesamples,
						__global float*  global_axial_result,
						__global float*  global_transverse_result){
	size_t global_id = get_global_id(0),i;
	float2 R1 = 0.0f, R2 = 0.0f;

	// Note: this averages across the end of a line to the next one. or even out of bounds.
	//for(i=0;i<(min(numb_avg,nlinesamples-(global_id % nlinesamples));i++){ // number to average over. 40=8/35*175
	for(i=0;i<numb_avg;i++){ // number to average over. 40=8/35*175
		float4 tmp = global_sum12_re_im[global_id*avg_offset+i];
		R1.x += tmp.x; // sum1_re
		R1.y += tmp.y; // sum1_im
		R2.x += tmp.z; // sum2_re
		R2.y += tmp.w; // sum2_im
	}
	R1 /= numb_avg;
	R2 /= numb_avg;

	// Don't care about this
	//global_axial_result[global_id]=k_axial*atan2(R1_im*R2_re-R2_im*R1_re,R1_re*R2_re+R1_im*R2_im);
	
	// data are arranged AXIAL,LATERAL_or_REPEAT,EMISSION, so modulo'ing by axial length will return sample depth.
	// global_id starts at zero, and is the scaling factor for k_trans for the wavenumber at depth
	global_transverse_result[global_id]=k_trans*(global_id % nlinesamples)*atan2(R1.y*R2.x+R2.y*R1.x,
																			     R1.x*R2.x-R1.y*R2.y);
}

/**	
 * This kernel finds the largest absolute value in a buffer
 * This is the first stage, the parallel reduction
 */
__kernel void maxabsval(__global float* buffer,
						__local float* scratch,
						__const int length,
						__global float* result) {
  int global_index = get_global_id(0); // What number of job am I? 66
  int global_size = get_global_size(0); //roundup(1136*75,64)
  int local_size = get_local_size(0); // How big is this group? 64
  int local_index = get_local_id(0); // Where am I in this group of 64?
  int group_id = get_group_id(0); // Which group of 64 am I in?

  float maxAbsVal = -INFINITY;
  // Loop sequentially over chunks of input vector, skip by 64s
  while (global_index < length) {
    float element = fabs(buffer[global_index]);
    maxAbsVal = (element > maxAbsVal) ? element : maxAbsVal;
    global_index += global_size;
  }

  // Check in our initial max, and wait for the other local threads to sync
  scratch[local_index] = maxAbsVal;
  barrier(CLK_LOCAL_MEM_FENCE);
  // Perform parallel reduction
  for(int offset = local_size / 2;
      offset > 0;
      offset /= 2) {
    if (local_index < offset) {
      float other = scratch[local_index + offset];
      float mine = scratch[local_index];
      scratch[local_index] = (mine > other) ? mine : other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_index == 0) {
    result[group_id] = scratch[0];
  }
}

/**	
 * This kernel finds the largest value in a 2 buffers
 * This is the second stage, loop across first stage results
 */
// ASK LEE IF THERE IS A MORE EFFICIENT SECOND STAGE HERE
__kernel void maxabsval2(__global float* result1,
						 __global float* result2,
						 __const  int    N,
						 __global float* maximum) {
	maximum[0]= -INFINITY;
	int i;
	for(i=0;i<N;i++) {
		maximum[0]= (result1[i] > maximum[0]) ? result1[i] : maximum[0];
		maximum[0]= (result2[i] > maximum[0]) ? result2[i] : maximum[0];
	}
}

/**	
 *	@param floatbufZ INPUT OpenCL buffer containing final velocity estimates for Z
 *	@param floatbufX INPUT OpenCL buffer containing final velocity estimates for X
 *	@param maximum
 *	@param scale 
 *	@param Nsamples
 *	@param outbufZ OUTPUT OpenCL buffer containing velocity estimates
 *	@param outbufX OUTPUT OpenCL buffer containing velocity estimates
 */
__kernel void combine(__global float* floatbufZ,
					  __global float* floatbufX,
					  __global float* maximum,
						const  float  scale,
					    const  int    Nsamples,
				      __global char*  outbufZ,
					  __global char*  outbufX) {
	size_t local_size = get_local_size(0), group_id = get_group_id(0),
		   local_id   = get_local_id(0),   global_id = get_global_id(0);
	const float a = 127.0*2./(3.1415927*scale);
	// FOR NOW, OMIT SCALING BY MAX
	//const float a = 127.0/maximum[0];

	if (global_id < Nsamples){
		// for the given point, normalize and copy the sample to the output buffer
		outbufZ[global_id] = convert_char(a*floatbufZ[global_id]);
		outbufX[global_id] = convert_char(a*floatbufX[global_id]);
	}
}
