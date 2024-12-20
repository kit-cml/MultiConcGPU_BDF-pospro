#include <cuda.h>
#include <cuda_runtime.h>

// #include "modules/drug_sim.hpp"
#include <math.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "modules/cipa_t.cuh"
#include "modules/glob_funct.hpp"
#include "modules/glob_type.hpp"
#include "modules/gpu.cuh"
namespace fs = std::filesystem;

#define ENOUGH ((CHAR_BIT * sizeof(int) - 1) / 3 + 3)
char buffer[255];

// unsigned int datapoint_size = 7000;
const unsigned int sample_limit = 10000;

clock_t START_TIMER;

clock_t tic();
void toc(clock_t start = START_TIMER);

clock_t tic() { return START_TIMER = clock(); }

void toc(clock_t start) {
    std::cout << "Elapsed time: " << (clock() - start) / (double)CLOCKS_PER_SEC << "s" << std::endl;
}

int gpu_check(unsigned int datasize) {
    int num_gpus;
    float percent;
    int id;
    size_t free, total;
    cudaGetDeviceCount(&num_gpus);
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        cudaSetDevice(gpu_id);
        cudaGetDevice(&id);
        cudaMemGetInfo(&free, &total);
        percent = (free / (float)total);
        printf("GPU No %d\nFree Memory: %ld, Total Memory: %ld (%f percent free)\n", id, free, total, percent * 100.0);
    }
    percent = 1.0 - (datasize / (float)total);
    //// this code strangely gave out too small value, so i disable the safety switch for now

    // printf("The program uses GPU No %d and %f percent of its memory\n", id,percent*100.0);
    // printf("\n");
    // if (datasize<=free) {
    //   return 0;
    // }
    // else {
    //   return 1;
    // }

    return 0;
}

// get the IC50 data from file
drug_t get_IC50_data_from_file(const char *file_name);
// return error and message based on the IC50 data

void addDrugData(char ***arrayOfStrings, int &size, const char newString[]) {
    char **newArray = new char *[size + 1];

    // Copy existing strings to the new array
    for (int i = 0; i < size; ++i) {
        newArray[i] = new char[strlen((*arrayOfStrings)[i]) + 1];
        strcpy(newArray[i], (*arrayOfStrings)[i]);
        delete[] (*arrayOfStrings)[i];  // Deallocate memory for old strings
    }

    // Allocate memory for the new string and copy it
    newArray[size] = new char[strlen(newString) + 1];
    strcpy(newArray[size], newString);

    // Deallocate memory for the old array
    delete[] *arrayOfStrings;

    // Update the pointer to point to the new array
    *arrayOfStrings = newArray;

    // Increment the size
    ++size;
}

int check_IC50_content(const drug_t *ic50, const param_t *p_param);

int get_IC50_data_from_file(const char *file_name, double *ic50) {
    /*
    a host function to take all samples from the file, assuming each sample has 14 features.

    it takes the file name, and an ic50 (already declared in 1D, everything become 1D)
    as a note, the data will be stored in 1D array, means this functions applies flatten.

    it returns 'how many samples were detected?' in integer.
    */
    FILE *fp_drugs;
    //   drug_t ic50;
    char *token;
    char buffer_ic50[255];
    unsigned int idx;

    if ((fp_drugs = fopen(file_name, "r")) == NULL) {
        printf("Cannot open file %s\n", file_name);
        return 0;
    }
    idx = 0;
    int sample_size = 0;
    fgets(buffer_ic50, sizeof(buffer_ic50), fp_drugs);                   // skip header
    while (fgets(buffer_ic50, sizeof(buffer_ic50), fp_drugs) != NULL) {  // begin line reading
        token = strtok(buffer_ic50, ",");
        while (token != NULL) {  // begin data tokenizing
            ic50[idx++] = strtod(token, NULL);
            token = strtok(NULL, ",");
        }  // end data tokenizing
        sample_size++;
    }  // end line reading

    fclose(fp_drugs);
    return sample_size;
}

int get_IC50_data_from_file(const char *file_name, double *ic50, double *conc, char **drug_name) {
    /*
    a host function to take all samples from the file, assuming each sample has 14 features.

    it takes the file name, and an ic50 (already declared in 1D, everything become 1D)
    as a note, the data will be stored in 1D array, means this functions applies flatten.

    it returns 'how many samples were detected?' in integer.
    */
    FILE *fp_drugs;
    //   drug_t ic50;
    char *token;
    char tmp_drug_name[32];
    char buffer_ic50[255];
    unsigned int idx_ic50, idx_conc;
    int drugsize = 0;

    if ((fp_drugs = fopen(file_name, "r")) == NULL) {
        printf("Cannot open file %s\n", file_name);
        return 0;
    }
    idx_ic50 = 0;
    idx_conc = 0;
    int sample_size = 0;
    fgets(buffer_ic50, sizeof(buffer_ic50), fp_drugs);                   // skip header
    while (fgets(buffer_ic50, sizeof(buffer_ic50), fp_drugs) != NULL) {  // begin line reading
        /*
        TODO: Extracting token from file
        1. take token for each file
        2. check the first token to drug_name, if already exist in array, then skip it
        3. check the second token to conc
        */

        token = strtok(buffer_ic50, ",");
        // printf("%s\n", token);  // testingAuto
        strcpy(tmp_drug_name, token);
        token = strtok(NULL, ",");
        // printf("%s\n", token);  // testingAuto
        strcat(tmp_drug_name, "_");
        strcat(tmp_drug_name, token);

        // printf("%s\n", tmp_drug_name);  // testingAuto
        addDrugData(&drug_name, drugsize, tmp_drug_name);
        conc[idx_conc++] = strtod(token, NULL);
        token = strtok(NULL, ",");
        // Check if there is wrong in here
        while (token != NULL) {  // begin data tokenizing
            ic50[idx_ic50++] = strtod(token, NULL);
            // printf("%s\n", token);  // testingAuto
            token = strtok(NULL, ",");
        }  // end data tokenizing
        sample_size++;
    }  // end line reading

    fclose(fp_drugs);
    return sample_size;
}

int get_cvar_data_from_file(const char *file_name, unsigned int limit, double *cvar) {
    // buffer for writing in snprintf() function
    char buffer_cvar[255];
    FILE *fp_cvar;
    // cvar_t cvar;
    char *token;
    // std::array<double,18> temp_array;
    unsigned int idx;

    if ((fp_cvar = fopen(file_name, "r")) == NULL) {
        printf("Cannot open file %s\n", file_name);
    }
    idx = 0;
    int sample_size = 0;
    fgets(buffer_cvar, sizeof(buffer_cvar), fp_cvar);                                              // skip header
    while ((fgets(buffer_cvar, sizeof(buffer_cvar), fp_cvar) != NULL) && (sample_size < limit)) {  // begin line reading
        token = strtok(buffer_cvar, ",");
        while (token != NULL) {  // begin data tokenizing
            cvar[idx++] = strtod(token, NULL);
            token = strtok(NULL, ",");
        }  // end data tokenizing
        // printf("\n");
        sample_size++;
        // cvar.push_back(temp_array);
    }  // end line reading

    fclose(fp_cvar);
    return sample_size;
}

int get_init_data_from_file(const char *file_name, double *init_states) {
    // buffer for writing in snprintf() function
    char buffer_cache[1023];
    FILE *fp_cache;
    // cvar_t cvar;
    char *token;
    // std::array<double,18> temp_array;
    unsigned long idx;

    if ((fp_cache = fopen(file_name, "r")) == NULL) {
        printf("Cannot open file %s\n", file_name);
    }
    idx = 0;
    unsigned int sample_size = 0;
    fgets(buffer_cache, sizeof(buffer_cache), fp_cache);                     // skip header
    while ((fgets(buffer_cache, sizeof(buffer_cache), fp_cache) != NULL)) {  // begin line reading
        token = strtok(buffer_cache, ",");
        while (token != NULL) {  // begin data tokenizing
            init_states[idx++] = strtod(token, NULL);
            // if(idx < 82){
            //     printf("%d: %lf\n",idx-1,init_states[idx-1]);
            // }
            token = strtok(NULL, ",");
        }  // end data tokenizing
        // printf("\n");
        sample_size++;
        // cvar.push_back(temp_array);
    }  // end line reading

    fclose(fp_cache);
    return sample_size;
}

int exists(const char *fname) {
    FILE *file;
    if ((file = fopen(fname, "r"))) {
        fclose(file);
        return 1;
    }
    // fclose(file);
    return 0;
}

int check_IC50_content(const drug_t *ic50, const param_t *p_param) {
    if (ic50->size() == 0) {
        printf("Something problem with the IC50 file!\n");
        return 1;
    } else if (ic50->size() > 2000) {
        printf("Too much input! Maximum sample data is 2000!\n");
        return 2;
    } else if (p_param->pace_max < 750 && p_param->pace_max > 1000) {
        printf("Make sure the maximum pace is around 750 to 1000!\n");
        return 3;
    }
    // else if(mympi::size > ic50->size()){
    // 	printf("%s\n%s\n",
    //               "Overflow of MPI Process!",
    //               "Make sure MPI Size is less than or equal the number of sample");
    // 	return 4;
    // }
    else {
        return 0;
    }
}

int get_herg_data_from_file(const char* file_name, double *herg)
{
  FILE *fp_herg;
  char *token;
  char buffer_herg[255];
  unsigned int idx;

  if( (fp_herg = fopen(file_name, "r")) == NULL){
    printf("Cannot open file %s\n", file_name);
    return 0;
  }
  printf("Herg file %s\n", file_name);
  idx = 0;
  int sample_size = 0;
  fgets(buffer_herg, sizeof(buffer_herg), fp_herg); // skip header
  while( fgets(buffer_herg, sizeof(buffer_herg), fp_herg) != NULL )
    { // begin line reading
      token = strtok( buffer_herg, "," );
      while( token != NULL )
      { // begin data tokenizing
        herg[idx++] = strtod(token, NULL);
        token = strtok(NULL, ",");
      } // end data tokenizing
      sample_size++;
    } // end line reading

  fclose(fp_herg);
  return sample_size;
}

int main(int argc, char **argv) {
    // enable real-time output in stdout
    setvbuf(stdout, NULL, _IONBF, 0);

    // NEW CODE STARTS HERE //
    double inal_auc_control = -90.547322;    // AUC of INaL under control model
    double ical_auc_control = -105.935067;   // AUC of ICaL under control model
    // mycuda *thread_id;
    // cudaMalloc(&thread_id, sizeof(mycuda));

    // input variables for cell simulation
    param_t *p_param, *d_p_param;
    p_param = new param_t();
    p_param->init();
    edison_assign_params(argc, argv, p_param);
    p_param->show_val();

    std::regex pattern("/([a-zA-Z0-9_\.]+)\.csv");
    std::smatch match;
    std::string fname = p_param->hill_file;
    std::regex_search(fname, match, pattern);

    printf("%s\n", match[1].str().c_str());

    double *ic50;  // temporary
    double *cvar;
    double *conc;
    char **drug_name = nullptr;

    ic50 = (double *)malloc(14 * sample_limit * sizeof(double));
    // if (p_param->is_cvar == true) cvar = (double *)malloc(18 * sample_limit * sizeof(double));
    cvar = (double *)malloc(18 * sample_limit * sizeof(double));
    conc = (double *)malloc(sample_limit * sizeof(double));
    double* herg = (double *)malloc(6 * sample_limit * sizeof(double));

    int num_of_constants = 208;
    int num_of_states = 49;
    int num_of_algebraic = 200;
    int num_of_rates = 49;

    // const double CONC = p_param->conc;

    //////// if we are in write time series mode (post processing) //////////

    printf("Using cached initial state from previous result!!!! \n\n");

    unsigned int datapoint_size;
    bool is_euler = true;

    if(is_euler){
        // datapoint_size = p_param->bcl / p_param->dt;
        datapoint_size = 1001;
    }
    else {
        datapoint_size = p_param->sampling_limit;
    }

    double *cache;
    cache = (double *)malloc(num_of_states * sample_limit * sizeof(double));

    double *d_ic50;
    double *d_conc;
    double *d_cvar;
    double *d_ALGEBRAIC;
    double *d_CONSTANTS;
    double *d_RATES;
    double *d_STATES;
    double *d_STATES_cache;

    // actually not used but for now, this is only for satisfiying the GPU regulator parameters
    double *d_STATES_RESULT;
    double *d_all_states;

    double *time;
    double *dt;
    double *states;
    double *ical;
    double *inal;
    double *cai_result;
    double *ina;
    double *ito;
    double *ikr;
    double *iks;
    double *ik1;
    cipa_t *temp_result, *cipa_result;

    static const int CALCIUM_SCALING = 1000000;
    static const int CURRENT_SCALING = 1000;

    // snprintf(buffer, sizeof(buffer),
    //   "./drugs/bepridil/IC50_samples.csv"
    //   // "./drugs/bepridil/IC50_optimal.csv"
    //   // "./IC50_samples.csv"
    //   );

    int sample_size = get_IC50_data_from_file(p_param->hill_file, ic50, conc, drug_name);
    if (sample_size == 0) printf("Something problem with the IC50 file!\n");
    // else if(sample_size > 2000)
    //     printf("Too much input! Maximum sample data is 2000!\n");
    printf("Sample size: %d\n", sample_size);
    printf("Set GPU Number: %d\n", p_param->gpu_index);

    cudaSetDevice(p_param->gpu_index);

    if (p_param->is_cvar == true) {
        int cvar_sample = get_cvar_data_from_file(p_param->cvar_file, sample_size, cvar);
        printf("Reading: %d Conductance Variability samples\n", cvar_sample);
    }

    int herg_size = get_herg_data_from_file(p_param->herg_file, herg);
        if(herg_size == 0)
            printf("Something problem with the herg file!\n");
        
        printf("herg size: %d herg check:\n", herg_size);
            for(int temp = 0; temp<6; temp++){
            printf("%lf, ",herg[temp]);
            } 
            printf("\n");

    printf("preparing GPU memory space \n");

    int cache_num = get_init_data_from_file(p_param->cache_file, cache);
    printf("Found cache for %d samples\n", cache_num);

    cudaMalloc(&d_ALGEBRAIC, num_of_algebraic * sample_size * sizeof(double));
    cudaMalloc(&d_CONSTANTS, num_of_constants * sample_size * sizeof(double));
    cudaMalloc(&d_RATES, num_of_rates * sample_size * sizeof(double));
    cudaMalloc(&d_STATES, num_of_states * sample_size * sizeof(double));
    cudaMalloc(&d_STATES_cache, (num_of_states + 2) * sample_size * sizeof(double));

    cudaMalloc(&d_p_param, sizeof(param_t));

    // prep for 1 cycle plus a bit (7000 * sample_size)
    cudaMalloc(&temp_result, sample_size * sizeof(cipa_t));
    cudaMalloc(&cipa_result, sample_size * sizeof(cipa_t));

    cudaMalloc(&time, sample_size * datapoint_size * sizeof(double));
    cudaMalloc(&dt, sample_size * datapoint_size * sizeof(double));
    cudaMalloc(&states, sample_size * datapoint_size * sizeof(double));
    cudaMalloc(&ical, sample_size * datapoint_size * sizeof(double));
    cudaMalloc(&inal, sample_size * datapoint_size * sizeof(double));
    cudaMalloc(&cai_result, sample_size * datapoint_size * sizeof(double));
    cudaMalloc(&ina, sample_size * datapoint_size * sizeof(double));
    cudaMalloc(&ito, sample_size * datapoint_size * sizeof(double));
    cudaMalloc(&ikr, sample_size * datapoint_size * sizeof(double));
    cudaMalloc(&iks, sample_size * datapoint_size * sizeof(double));
    cudaMalloc(&ik1, sample_size * datapoint_size * sizeof(double));
    // cudaMalloc(&d_STATES_RESULT, (num_of_states+1) * sample_size * sizeof(double));
    // cudaMalloc(&d_all_states, num_of_states * sample_size * p_param->find_steepest_start * sizeof(double));

    printf("Copying sample files to GPU memory space \n");
    cudaMalloc(&d_ic50, sample_size * 14 * sizeof(double));
    cudaMalloc(&d_cvar, sample_size * 18 * sizeof(double));
    cudaMalloc(&d_conc, sample_size * sizeof(double));
    
    cudaMemcpy(d_STATES_cache, cache, (num_of_states + 2) * sample_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ic50, ic50, sample_size * 14 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cvar, cvar, sample_size * 18 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conc, conc, sample_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p_param, p_param, sizeof(param_t), cudaMemcpyHostToDevice);

      // for BDF
        //double *d_all_states;
        double *d_herg;

        double *y; double *y_new; double *F; double *delta; double *Jc; 
        double *y_perturbed; double *g0; double *g_perturbed; 
        cudaMalloc(&y, num_of_states * sample_size * sizeof(double));
        cudaMalloc(&y_new, num_of_states * sample_size * sizeof(double));
        cudaMalloc(&F, num_of_states * sample_size * sizeof(double));
        cudaMalloc(&delta, num_of_states * sample_size * sizeof(double));
        cudaMalloc(&Jc, num_of_states * num_of_states * sample_size * sizeof(double));

        cudaMalloc(&y_perturbed, num_of_states * sample_size * sizeof(double));
        cudaMalloc(&g0, num_of_states * sample_size * sizeof(double));
        cudaMalloc(&g_perturbed, num_of_states * sample_size * sizeof(double));

        cudaMalloc(&d_all_states, num_of_states * sample_size * p_param->find_steepest_start * sizeof(double)); // for each sample
        cudaMalloc(&d_herg, 6 * sample_size * sizeof(double));
        cudaMemcpy(d_herg, herg, sample_size * 6 * sizeof(double), cudaMemcpyHostToDevice);

    // // Get the maximum number of active blocks per multiprocessor
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, do_drug_sim_analytical, threadsPerBlock);

    // // Calculate the total number of blocks
    // int numTotalBlocks = numBlocks * cudaDeviceGetMultiprocessorCount();

    tic();
    printf("Timer started, doing simulation.... \n\n\nGPU Usage at this moment: \n");
    int thread = 20;
    int block = (sample_size + thread - 1) / thread;
    // int block = (sample_size + thread - 1) / thread;
    if (gpu_check(15 * sample_size * sizeof(double) + sizeof(param_t)) == 1) {
        printf("GPU memory insufficient!\n");
        return 0;
    }
    printf("Sample size: %d\n", sample_size);
    cudaSetDevice(p_param->gpu_index);
    printf("\n   Configuration: \n\n\tblock\t||\tthread\n---------------------------------------\n  \t%d\t||\t%d\n\n\n",
           block, thread);
    // initscr();
    // printf("[____________________________________________________________________________________________________]
    // 0.00 %% \n");
    // (double *d_ic50, double *d_cvar, double *d_conc, double *d_CONSTANTS,
    //                                   double *d_STATES, double *d_STATES_cache, double *d_RATES, double *d_ALGEBRAIC,
    //                                   double *d_STATES_RESULT, double *d_all_states, double *d_herg,
    //                                   double *time, double *states,
    //                                   double *out_dt, double *cai_result, double *ina, double *inal, double *ical,
    //                                   double *ito, double *ikr, double *iks, double *ik1, unsigned int sample_size,
    //                                   cipa_t *temp_result, cipa_t *cipa_result, param_t *p_param,
    //                                   double *y, double *y_new, double *F, double *delta, double *Jc, double *y_perturbed, double *g0, double *g_perturbed) 

    kernel_DrugSimulation<<<block, thread>>>(d_ic50, d_cvar, d_conc, d_CONSTANTS, d_STATES, d_STATES_cache, d_RATES,
                                             d_ALGEBRAIC, d_STATES_RESULT, d_all_states, d_herg, time, states, dt, cai_result,
                                             ina, inal, ical, ito, ikr, iks, ik1, sample_size, temp_result, cipa_result,
                                             d_p_param, y, y_new, F, delta, Jc, y_perturbed, g0, g_perturbed);
    // block per grid, threads per block
    // endwin();

    cudaDeviceSynchronize();

    printf("allocating memory for computation result in the CPU, malloc style \n");
    double *h_states, *h_time, *h_dt, *h_ical, *h_inal, *h_cai_result, *h_ina, *h_ito, *h_ikr, *h_iks, *h_ik1;
    cipa_t *h_cipa_result;

    h_states = (double *)malloc(datapoint_size * sample_size * sizeof(double));
    printf("...allocated for STATES, \n");
    h_time = (double *)malloc(datapoint_size * sample_size * sizeof(double));
    printf("...allocated for time, \n");
    h_dt = (double *)malloc(datapoint_size * sample_size * sizeof(double));
    printf("...allocated for dt, \n");
    h_cai_result = (double *)malloc(datapoint_size * sample_size * sizeof(double));
    printf("...allocated for Cai, \n");
    h_ina = (double *)malloc(datapoint_size * sample_size * sizeof(double));
    printf("...allocated for iNa, \n");
    h_ito = (double *)malloc(datapoint_size * sample_size * sizeof(double));
    printf("...allocated for ito, \n");
    h_ikr = (double *)malloc(datapoint_size * sample_size * sizeof(double));
    printf("...allocated for ikr, \n");
    h_iks = (double *)malloc(datapoint_size * sample_size * sizeof(double));
    printf("...allocated for iks, \n");
    h_ik1 = (double *)malloc(datapoint_size * sample_size * sizeof(double));
    printf("...allocated for ik1, \n");
    h_ical = (double *)malloc(datapoint_size * sample_size * sizeof(double));
    printf("...allocated for ICaL, \n");
    h_inal = (double *)malloc(datapoint_size * sample_size * sizeof(double));

    h_cipa_result = (cipa_t *)malloc(sample_size * sizeof(cipa_t));
    printf("...allocating for INaL and postprocessing, all set!\n");

    ////// copy the data back to CPU, and write them into file ////////
    printf("copying the data back to the CPU \n");

    cudaMemcpy(h_states, states, sample_size * datapoint_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_time, time, sample_size * datapoint_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dt, dt, sample_size * datapoint_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ical, ical, sample_size * datapoint_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_inal, inal, sample_size * datapoint_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cai_result, cai_result, sample_size * datapoint_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ina, ina, sample_size * datapoint_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ito, ito, sample_size * datapoint_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ikr, ikr, sample_size * datapoint_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_iks, iks, sample_size * datapoint_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ik1, ik1, sample_size * datapoint_size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_cipa_result, cipa_result, sample_size * sizeof(cipa_t), cudaMemcpyDeviceToHost);

    cudaFree(d_ALGEBRAIC);
    cudaFree(d_CONSTANTS);
    cudaFree(d_RATES);
    cudaFree(d_STATES);
    cudaFree(d_p_param);
    cudaFree(temp_result);
    cudaFree(cipa_result);
    cudaFree(d_STATES_RESULT);
    cudaFree(d_ic50);
    cudaFree(d_cvar);
    cudaFree(d_conc);
    cudaFree(time);
    cudaFree(dt);
    cudaFree(states);
    cudaFree(ical);
    cudaFree(inal);
    cudaFree(cai_result);
    cudaFree(ina);
    cudaFree(ito);
    cudaFree(ikr);
    cudaFree(iks);
    cudaFree(ik1);

    FILE *writer;
    int check;
    bool folder_created = false;

    printf("writing to file... \n");
    // sample loop
    for (int sample_id = 0; sample_id < sample_size; sample_id++) {
        // printf("writing sample %d... \n",sample_id);
        char sample_str[ENOUGH];
        char conc_str[ENOUGH];
        char filename[500] = "./result/post_";
        sprintf(sample_str, "%d", sample_id);
        // sprintf(conc_str, "%.2f", conc[sample_id]);
        strcat(filename, match[1].str().c_str());
        strcat(filename, "/");
        if (folder_created == false) {
            check = mkdir(filename, 0777);
            // check if directory is created or not
            if (!check) {
                printf("Directory created\n");
            } else {
                printf("Unable to create directory, or the folder is already created, relax mate...\n");
            }
            folder_created = true;
        }

        strcat(filename, sample_str);
        strcat(filename, "_pace.csv");

        writer = fopen(filename, "w");
        fprintf(writer, "Time,Vm,dVm/dt,Cai,INa,INaL,ICaL,IKs,IKr,IK1,Ito\n");
        for (int datapoint = 1; datapoint < datapoint_size; datapoint++) {
            if (h_time[sample_id + (datapoint * sample_size)] == 0.0) {
                break;
            }
            fprintf(writer,
                    "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",  // change this into string, or limit the decimal
                                                                      // accuracy, so we can decrease filesize
                    h_time[sample_id + (datapoint * sample_size)], h_states[sample_id + (datapoint * sample_size)],
                    h_dt[sample_id + (datapoint * sample_size)], h_cai_result[sample_id + (datapoint * sample_size)],

                    h_ina[sample_id + (datapoint * sample_size)], h_inal[sample_id + (datapoint * sample_size)],

                    h_ical[sample_id + (datapoint * sample_size)], h_iks[sample_id + (datapoint * sample_size)],

                    h_ikr[sample_id + (datapoint * sample_size)], h_ik1[sample_id + (datapoint * sample_size)],

                    h_ito[sample_id + (datapoint * sample_size)]);
        }
        fclose(writer);
    }

    printf("writing each biomarkers value... \n");
    // sample loop
    // char conc_str[ENOUGH];
    char filename[500] = "./result/post_";
    // sprintf(sample_str, "%d", sample_id);
    // sprintf(conc_str, "%.2f", conc[sample_id]);
    strcat(filename, match[1].str().c_str());
    strcat(filename, "/");
    // printf("creating %s... \n", filename);
    if (folder_created == false) {
        check = mkdir(filename, 0777);
        // check if directory is created or not
        if (!check) {
            printf("Directory created\n");
        } else {
            printf("Unable to create directory, or the folder is already created, relax mate...\n");
        }
        folder_created = true;
    }

    // strcat(filename,sample_str);
    strcat(filename, "_biomarkers.csv");

    writer = fopen(filename, "a");

    fprintf(writer,"sample,qnet,inal_auc,ical_auc,apd90,apd50,apd_tri,cad90,cad50,cad_tri,dvmdt_repol,dvmdt_peak,vm_peak,vm_valley,vm_dia,ca_peak,ca_valley,ca_dia\n");
    for (int sample_id = 0; sample_id < sample_size; sample_id++) {
        // printf("writing sample %d... \n",sample_id);

        fprintf(writer,
                "%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",  // change this into string, or
                                                                                         // limit the decimal accuracy,
                                                                                         // so we can decrease filesize
                sample_id, h_cipa_result[sample_id].qnet, h_cipa_result[sample_id].inal_auc,
                h_cipa_result[sample_id].ical_auc,

                h_cipa_result[sample_id].apd90, h_cipa_result[sample_id].apd50,
                h_cipa_result[sample_id].apd90 - h_cipa_result[sample_id].apd50,

                h_cipa_result[sample_id].cad90, h_cipa_result[sample_id].cad50,
                h_cipa_result[sample_id].cad90 - h_cipa_result[sample_id].cad50,

                h_cipa_result[sample_id].dvmdt_repol, h_cipa_result[sample_id].dvmdt_max, h_cipa_result[sample_id].vm_peak,
                h_cipa_result[sample_id].vm_valley, h_cipa_result[sample_id].vm_dia,

                h_cipa_result[sample_id].ca_peak, h_cipa_result[sample_id].ca_valley, 
                h_cipa_result[sample_id].ca_dia
                

        );
    }
    fclose(writer);

    toc();

    return 0;
}
