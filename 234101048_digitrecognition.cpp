// 234101048_digitrecognition.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include<stdio.h>
#include<string.h>
#include<limits.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<float.h>
#include<Windows.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <limits>
#include <cmath>
#include <fstream>
#include <cstring>
#include <windows.h>

#define K 32					//LBG Codebook Size
#define DELTA 0.0001			//K-Means Parameter
#define EPSILON 0.03			 //LBG Splitting Parameter
#define UNIVERSE_SIZE 50000		//Universe Size
#define CLIP 5000				//Max value after normalizing
#define FS 320					//Frame Size
#define Q 12					//No. of cepstral coefficient
#define P 12					//No. of LPC
#define pie (22.0/7)
#define N 5						//no. of states in HMM Model
#define M 32					//Codebook Size
#define T_ 400					//Max possible no. of frames
#define TRAIN_SIZE 20			//Training Files for each utterance


//HMM Model Variables
long double A[N + 1][N + 1],B[N + 1][M + 1], pi[N + 1], alpha[T_ + 1][N + 1], beta[T_ + 1][N + 1], gamma[T_ + 1][N + 1], delta[T_+1][N+1], xi[T_+1][N+1][N+1], A_bar[N + 1][N + 1],B_bar[N + 1][M + 1], pi_bar[N + 1];
int O[T_+1], q[T_+1], psi[T_+1][N+1], q_star[T_+1];
long double P_star=-1, P_star_dash=-1;

//Store 1 file values
int samples[50000];
//No. of frames in file
int T=160;
//Index of start and end markers
int start_frame;
int end_frame;

//Durbin's Algo variables
long double R[P+1];
long double a[P+1];
//Cepstral Coefficient
long double C[Q+1];
//Store codebook
long double reference[M+1][Q+1];
//Tokhura Weights
long double tokhuraWeight[Q+1]={0.0, 1.0, 3.0, 7.0, 13.0, 19.0, 22.0, 25.0, 33.0, 42.0, 50.0, 56.0, 61.0};
//Store energry per frame
long double energy[T_]={0};
//Universe vector
long double X[UNIVERSE_SIZE][Q];
//Universe Vector size
int LBG_M=0;
//Codebook
long double codebook[K][Q];
//Store mapping of universe with cluster
int cluster[UNIVERSE_SIZE];


const int ROWS = 6;
const int COLS = 7;

void create_board(int board[ROWS][COLS]) {
    memset(board, 0, ROWS * COLS * sizeof(int));
}

void drop_piece(int board[ROWS][COLS], int row, int col, int piece) {
    board[row][col] = piece;
}

bool is_valid_location(const int board[ROWS][COLS], int col) {
    return board[ROWS - 1][col] == 0;
}

int get_next_open_row(const int board[ROWS][COLS], int col) {
    for (int r = 0; r < ROWS; r++) {
        if (board[r][col] == 0) {
            return r;
        }
    }
    return -1;
}

bool winning_move(const int board[ROWS][COLS], int piece) {
    // Check horizontal locations for win
    for (int c = 0; c < COLS - 3; c++) {
        for (int r = 0; r < ROWS; r++) {
            if (board[r][c] == piece && board[r][c + 1] == piece &&
                board[r][c + 2] == piece && board[r][c + 3] == piece) {
                return true;
            }
        }
    }
    // Check vertical locations for win
    for (int c = 0; c < COLS; c++) {
        for (int r = 0; r < ROWS - 3; r++) {
            if (board[r][c] == piece && board[r + 1][c] == piece &&
                board[r + 2][c] == piece && board[r + 3][c] == piece) {
                return true;
            }
        }
    }
    // Check positively sloped diagonals
    for (int c = 0; c < COLS - 3; c++) {
        for (int r = 0; r < ROWS - 3; r++) {
            if (board[r][c] == piece && board[r + 1][c + 1] == piece &&
                board[r + 2][c + 2] == piece && board[r + 3][c + 3] == piece) {
                return true;
            }
        }
    }
    // Check negatively sloped diagonals
    for (int c = 0; c < COLS - 3; c++) {
        for (int r = 3; r < ROWS; r++) {
            if (board[r][c] == piece && board[r - 1][c + 1] == piece &&
                board[r - 2][c + 2] == piece && board[r - 3][c + 3] == piece) {
                return true;
            }
        }
    }
    return false;
}

int score_position(const int board[ROWS][COLS], int piece) {
    // Simple heuristic function to evaluate the board
    int score = 0;
    // Score center column
    int center_count = 0;
    for (int r = 0; r < ROWS; r++) {
        if (board[r][3] == piece) {
            center_count++;
        }
    }
    score += center_count * 3;
    return score;
}

void minimax(int board[ROWS][COLS], int depth, int alpha, int beta, bool maximizingPlayer, int result[2]) {
    int valid_locations[COLS];
    int valid_count = 0;
    for (int c = 0; c < COLS; c++) {
        if (is_valid_location(board, c)) {
            valid_locations[valid_count++] = c;
        }
    }
    bool is_terminal = valid_count == 0;
    if (depth == 0 || is_terminal) {
        result[0] = -1;
        result[1] = (is_terminal) ? 0 : score_position(board, 2);
        return;
    }
    if (maximizingPlayer) {
        int value = -2147483648;
        int column = valid_locations[rand() % valid_count];
        for (int i = 0; i < valid_count; i++) {
            int col = valid_locations[i];
            int row = get_next_open_row(board, col);
            int b_copy[ROWS][COLS];
            memcpy(b_copy, board, ROWS * COLS * sizeof(int));
            drop_piece(b_copy, row, col, 2);
            int new_result[2];
            minimax(b_copy, depth - 1, alpha, beta, false, new_result);
            int new_score = new_result[1];
            if (new_score > value) {
                value = new_score;
                column = col;
            }
            alpha = max(alpha, value);
            if (alpha >= beta) {
                break;
            }
        }
        result[0] = column;
        result[1] = value;
    } else {
        int value = 2147483647;
        int column = valid_locations[rand() % valid_count];
        for (int i = 0; i < valid_count; i++) {
            int col = valid_locations[i];
            int row = get_next_open_row(board, col);
            int b_copy[ROWS][COLS];
            memcpy(b_copy, board, ROWS * COLS * sizeof(int));
            drop_piece(b_copy, row, col, 1);
            int new_result[2];
            minimax(b_copy, depth - 1, alpha, beta, true, new_result);
            int new_score = new_result[1];
            if (new_score < value) {
                value = new_score;
                column = col;
            }
            beta = min(beta, value);
            if (alpha >= beta) {
                break;
            }
        }
        result[0] = column;
        result[1] = value;
    }
}


void print_board(const int board[ROWS][COLS]) {
    printf("  ");
    for (int i = 0; i < COLS; ++i) {
        printf("%d  ", i);
    }
    printf("\n------------------------\n");
    for (int r = ROWS - 1; r >= 0; --r) {
        printf("| ");
        for (int c = 0; c < COLS; ++c) {
            if (board[r][c] == 0) {
                printf(" . ");
            } else if (board[r][c] == 1) {
                printf(" X ");
            } else {
                printf(" O ");
            }
        }
        printf("|\n");
    }
    printf("------------------------\n");
}

//Normalize the data
void normalize_data(char file[100]){
	//open inputfile
	FILE* fp=fopen(file,"r");
	if(fp==NULL){
		printf("Error in Opening File!\n");
		return;
	}
	int err1;
	int amp=0,avg=0;
	int i=0;
	int n=0;
	int min_amp=INT_MAX;
	int max_amp=INT_MIN;
	//calculate average, minimum & maximum amplitude
	for(int m=0;m<10;m++)
		{
 			err1 = fscanf_s( fp, "%*s");
		}
	while(!feof(fp)){
		fscanf(fp,"%d",&amp);
		avg+=amp;
		min_amp=(amp<min_amp)?amp:min_amp;
		max_amp=(amp>max_amp)?amp:max_amp;
		n++;
	}
	avg/=n;
	T=(n-FS)/80 + 1;
	if(T>T_) T=T_;
	//update minimum & maximum amplitude after DC Shift
	min_amp-=avg;
	max_amp-=avg;
	fseek(fp,0,SEEK_SET);
	for(int m=0;m<10;m++)
		{
 			err1 = fscanf_s( fp, "%*s");
		}
	while(!feof(fp)){
		fscanf(fp,"%d",&amp);
		if(min_amp==max_amp){
			amp=0;
		}
		else{
			//handle DC Shift
			amp-=avg;
			//normalize the data
			amp=(amp*CLIP)/((max_amp>min_amp)?max_amp:(-1)*min_amp);
			//store normalized data
			samples[i++]=amp;
		}
	}
	fclose(fp);
}

//calculate energy of frame
void calculate_energy_of_frame(int frame_no){
	int sample_start_index=frame_no*80;
	energy[frame_no]=0;
	for(int i=0;i<FS;i++){
		energy[frame_no]+=samples[i+sample_start_index]*samples[i+sample_start_index];
		energy[frame_no]/=FS;
	}
}

//Calculate Max Energy of file
long double calculate_max_energy(){
	int nf=T;
	long double max_energy=DBL_MIN;
	for(int f=0;f<nf;f++){
		if(energy[f]>max_energy){
			max_energy=energy[f];
		}
	}
	return max_energy;
}

//calculate average energy of file
long double calculate_avg_energy(){
	int nf=T;
	long double avg_energy=0.0;
	for(int f=0;f<nf;f++){
		avg_energy+=energy[f];
	}
	return avg_energy/nf;
}

//mark starting and ending of speech activity
void mark_checkpoints(){
	int nf=T;
	//Calculate energy of each frame
	for(int f=0;f<nf;f++){
		calculate_energy_of_frame(f);
	}
	//Make 10% of average energy as threshold
	long double threshold_energy=calculate_avg_energy()/10;
	//long double threshold_energy=calculate_max_energy()/10;
	int isAboveThresholdStart=1;
	int isAboveThresholdEnd=1;
	start_frame=0;
	end_frame=nf-1;
	//Find start frame where speech activity starts
	for(int f=0;f<nf-5;f++){
		for(int i=0;i<5;i++){
			isAboveThresholdStart*=(energy[f+i]>threshold_energy);
		}
		if(isAboveThresholdStart){
			start_frame=((f-5) >0)?(f-5):(0);
			break;
		}
		isAboveThresholdStart=1;
	}
	//Find end frame where speech activity ends
	for(int f=nf-1;f>4;f--){
		for(int i=0;i<5;i++){
			isAboveThresholdEnd*=(energy[f-i]>threshold_energy);
		}
		if(isAboveThresholdEnd){
			end_frame=((f+5) < nf)?(f+5):(nf-1);
			break;
		}
		isAboveThresholdEnd=1;
	}
}

//load codebook
void load_codebook(){
	FILE* fp;
	//fp=fopen("digit_codebook.csv","r");
	fp=fopen("234101048_codebook.csv","r");
	if(fp==NULL){
		printf("Error in Opening File!\n");
		return;
	}
	for(int i=1;i<=M;i++){
		for(int j=1;j<=Q;j++){
			fscanf(fp,"%Lf,",&reference[i][j]);
		}
	}
	fclose(fp);
}

//Calculate Ai's using Levenson Durbin Algorithm
void durbinAlgo(){
	
	long double E=R[0];
	long double alpha[13][13];
	for(int i=1;i<=P;i++){
		double k;
		long double numerator=R[i];
		long double alphaR=0.0;
		for(int j=1;j<=(i-1);j++){
			alphaR+=alpha[j][i-1]*R[i-j];
		}
		numerator-=alphaR;
		k=numerator/E;
		alpha[i][i]=k;
		
		for(int j=1;j<=(i-1);j++){
			alpha[j][i]=alpha[j][i-1]-(k*alpha[i-j][i-1]);
			if(i==P){
				a[j]=alpha[j][i];
			}
		}
		E=(1-k*k)*E;
		if(i==P){
			a[i]=alpha[i][i];
		}
	}
}

//Calculate minimun LPC Coefficients using AutoCorrelation
void autoCorrelation(int frame_no){
	long double s[FS];
	int sample_start_index=frame_no*80;
	
	//Hamming Window Function
	for(int i=0;i<FS;i++){
		long double wn=0.54-0.46*cos((2*(22.0/7.0)*i)/(FS-1));
		s[i]=wn*samples[i+sample_start_index];
	}
	
	//Calculate R0 to R12
	for(int i=0;i<=P;i++){
		long double sum=0.0;
		for(int y=0;y<=FS-1-i;y++){
			sum+=((s[y])*(s[y+i]));
		}
		R[i]=sum;
	}

	//Apply Durbin's Algorithm to calculate ai's
	durbinAlgo();
}


//Find Cepstral Coefficients
void cepstralTransformation(){
	C[0]=2.0*(log(R[0])/log(2.0));
	for(int m=1;m<=P;m++){
		C[m]=a[m];
		for(int k=1;k<m;k++){
			C[m]+=((k*C[k]*a[m-k])/m);
		}
	}
}

//Apply Raised Sine Window on Cepstral Coefficients
void raisedSineWindow(){
	for(int m=1;m<=P;m++){
		long double wm=(1+(Q/2)*sin(pie*m/Q));
		C[m]*=wm;
	}
}

//Store Cepstral coefficients of each frame of file
void process_universe_file(FILE* fp, char file[]){
	//normalize data
	normalize_data(file);
	int m=0;
	int nf=T;
	//repeat procedure for frames
	for(int f=0;f<nf;f++){
		//Apply autocorrelation
		autoCorrelation(f);
		//Apply cepstral Transformation
		cepstralTransformation();
		//apply raised sine window
		raisedSineWindow();
		for(int i=1;i<=Q;i++){
			fprintf(fp,"%Lf,",C[i]);
		}
		fprintf(fp,"\n");
	}
}

//Generate Universe from given dataset
void generate_universe(){
	FILE* universefp;
	universefp=fopen("234101048_universe.csv","w");
	for(int d=0;d<=6;d++){
		for(int u=1;u<=TRAIN_SIZE;u++){
			char filename[40];
			_snprintf(filename,40,"234101048_dataset/234101048_E_%d_%d.txt",d,u);
			process_universe_file(universefp,filename);
		}
	}
	
}

//calculate minimium Tokhura Distance
int minTokhuraDistance(long double testC[]){
	long double minD=DBL_MAX;
	int minDi=0;
	for(int i=1;i<=M;i++){
		long double distance=0.0;
		for(int j=1;j<=Q;j++){
			distance+=(tokhuraWeight[j]*(testC[j]-reference[i][j])*(testC[j]-reference[i][j]));
		}
		if(distance<minD){
			minD=distance;
			minDi=i;
		}
	}
	return minDi;
}

//Generate Observation Sequence
void generate_observation_sequence(char file[]){
	FILE* fp=fopen("o.txt","w");
	//normalize data
	normalize_data(file);
	int m=0;
	//mark starting and ending index
	mark_checkpoints();
	T=(end_frame-start_frame+1);
	int nf=T;
	//long double avg_energy=calculate_avg_energy();
	//repeat procedure for each frames
	for(int f=start_frame;f<=end_frame;f++){
		//Apply autocorrelation
		autoCorrelation(f);
		//Apply cepstral Transformation
		cepstralTransformation();
		//apply raised sine window "or" liftering
		raisedSineWindow();
		fprintf(fp,"%d ",minTokhuraDistance(C));
	}
	fprintf(fp,"\n");
	fclose(fp);
}



void load_universe(char file[100]){
	//open inputfile
	FILE* fp=fopen(file,"r");
	if(fp==NULL){
		printf("Error in Opening File!\n");
		return;
	}
	
	int i=0;
	long double c;
	while(!feof(fp)){
		fscanf(fp,"%Lf,",&c);
		X[LBG_M][i]=c;
		//Ceptral coeffecient index
		i=(i+1)%12;
		//Compute Universe vector size
		if(i==0) LBG_M++;
	}
	fclose(fp);
}

void store_codebook(char file[100],int k){
	FILE* fp=fopen(file,"w");
	if(fp==NULL){
		printf("Error opening file!\n");
		return;
	}
	for(int i=0;i<k;i++){
		for(int j=0;j<12;j++){
			fprintf(fp,"%Lf,",codebook[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}



void print_codebook(int k){
	//printf(" Codebook of size %d:\n",k);
	for(int i=0;i<k;i++){
		for(int j=0;j<12;j++){
			if(j==0){//printf(" %Lf",codebook[i][j]);
			}
			else if(codebook[i][j]<0)
			{
		//		printf("  %Lf",codebook[i][j]);
			}
			else
			{
			//	printf("   %Lf",codebook[i][j]);
			}
			
		}

	//	printf("\n");
	}
	//printf("\n");
}


void initialize_with_centroid(){
	long double centroid[12]={0.0};
	for(int i=0;i<LBG_M;i++){
		for(int j=0;j<12;j++){
			centroid[j]+=X[i][j];
		}
	}
	for(int i=0;i<12;i++){
		centroid[i]/=LBG_M;
		codebook[0][i]=centroid[i];
	}
	print_codebook(1);
}


long double calculate_distance(long double x[12], long double y[12]){
	long double distance=0.0;
	for(int i=0;i<12;i++){
		distance+=(tokhuraWeight[i+1]*(x[i]-y[i])*(x[i]-y[i]));
	}
	return distance;
}



void nearest_neighbour(int k){
	for(int i=0;i<LBG_M;i++){
		//store minimum distance between input and codebook
		long double nn=DBL_MAX;
		//store index of codevector with which input has minimum distance
		int cluster_index;
		for(int j=0;j<k;j++){
			//compute distance between input and codevector
			long double dxy=calculate_distance(X[i],codebook[j]);
			if(dxy<=nn){
				cluster_index=j;
				nn=dxy;
			}
		}
		//classification of ith input to cluster_index cluster
		cluster[i]=cluster_index;
	}
}


void codevector_update(int k){
	long double centroid[K][12]={0.0};
	//Store number of vectors in each cluster
	int n[K]={0};
	for(int i=0;i<LBG_M;i++){
		for(int j=0;j<12;j++){
			centroid[cluster[i]][j]+=X[i][j];
		}
		n[cluster[i]]++;
	}
	//Codevector Updation as Centroid of each cluster
	for(int i=0;i<k;i++){
		for(int j=0;j<12;j++){
			codebook[i][j]=centroid[i][j]/n[i];
		}
	}
}


long double calculate_distortion(){
	long double distortion=0.0;
	for(int i=0;i<LBG_M;i++){
		distortion+=calculate_distance(X[i],codebook[cluster[i]]);
	}
	distortion/=LBG_M;
	return distortion;
}


void KMeans(int k){
	FILE* fp=fopen("distortion.txt","a");
	if(fp==NULL){
		printf("Error opening file!\n");
		return;
	}
	//iterative index
	int m=0;
	//store previous and current D
	long double prev_D=DBL_MAX, cur_D=DBL_MAX;
	//repeat until convergence
	do{
		//Classification
		nearest_neighbour(k);
		//Iterative index update
		m++;
		//Codevector Updation
		codevector_update(k);
		prev_D=cur_D;
		//Calculate overall avg Distortion / D
		cur_D=calculate_distortion();
		//printf(" Iteration: %d\t\t",m);
		//printf("Distortion : %Lf\n",cur_D);
		fprintf(fp,"%Lf\n",cur_D);
	}while((prev_D-cur_D)>DELTA);//repeat until distortion difference is >delta
	//Print Updated Codebook
	//printf("\n Updated");
	print_codebook(k);
	fclose(fp);
}



void LBG(){
	
	//Start from single codebook
	int k=1;
	//Compute codevector as centroid of universe
	initialize_with_centroid();
	//repeat until desired size codebook is reached
	while(k!=K){
		//Split each codebook entry Yi to Yi(1+epsilon) & Yi(1-epsilon)
		for(int i=0;i<k;i++){
			for(int j=0;j<12;j++){
				long double Yi=codebook[i][j];
				//Yi(1+epsilon)
				codebook[i][j]=Yi-EPSILON;
				//Yi(1-epsilon)
				codebook[i+k][j]=Yi+EPSILON;
			}
		}
		//Double size of codebook
		k=k*2;
		//Call K-means with split codebook
		KMeans(k);
	}
}

//Generate Codebook From given Universe
void generate_codebook(){
	load_universe("234101048_universe.csv");

	LBG();
	store_codebook("234101048_codebook.csv",K);
}


//Initialize every variable of HMM module to zero
void initialization()
{
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			A[i][j] = 0;
		}
		for (int j = 1; j <= M; j++)
		{
			B[i][O[j]] = 0;
		}
		pi[i] = 0;
	}
	for (int i = 1; i <= T; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			alpha[i][j] = 0;
			beta[i][j] = 0;
			gamma[i][j] = 0;
		}
	}
}

//Calculate Alpha
//Forward Procedure
void calculate_alpha()
{
	//Initialization
	for (int i = 1; i <= N; i++)
	{
		alpha[1][i] = pi[i] * B[i][O[1]];
	}
	//Induction
	for (int t = 1; t < T; t++)
	{
		for (int j = 1; j <= N; j++)
		{
			long double sum = 0;
			for (int i = 1; i <= N; i++)
			{
				sum += alpha[t][i] * A[i][j];
			}
			alpha[t + 1][j] = sum * B[j][O[t + 1]];
		}
	}

	//Store Alpha in File
	FILE *fp=fopen("alpha.txt","w");
	//printf("\nAlpha Matrix\n");
	for (int t = 1; t <= T; t++)
	{
		for (int j = 1; j <= N; j++)
		{
			//printf("%e ", alpha[t][j]);
			fprintf(fp,"%e\t", alpha[t][j]);
		}
		//printf("\n");
		fprintf(fp,"\n");
	}
	fclose(fp);
	//printf("\n\n");
}

//Solution to problem1; Evaluate model
//P(O|lambda)=sigma_i=1toN(alpha[T][i])
long double calculate_score()
{
	long double probability = 0;
	for (int i = 1; i <= N; i++)
	{
		probability += alpha[T][i];
	}
	//printf("Probability P(O/lambda)= %.16e\n", probability);
	return probability;
}

//Calculate Beta
//Backward Procedure
void calculate_beta()
{
	//Initailization
	for (int i = 1; i <= N; i++)
	{
		beta[T][i] = 1;
	}
	//Induction
	for (int t = T - 1; t >= 1; t--)
	{
		for (int i = 1; i <= N; i++)
		{
			for (int j = 1; j <= N; j++)
			{
				beta[t][i] += A[i][j] * B[j][O[t + 1]] * beta[t + 1][j];
			}
		}
	}
	//Store beta values in file
	FILE *fp=fopen("beta.txt","w");
	//printf("Beta Matrix\n");
	for (int t = 1; t < T; t++)
	{
		for (int j = 1; j <= N; j++)
		{
			//printf("%.16e ", beta[t][j]);
			fprintf(fp,"%e\t", beta[t][j]);
		}
		//printf("\n");
		fprintf(fp,"\n");
	}
	fclose(fp);
	//printf("\n\n");
}

//Predict most individually likely states using gamma
//One of the solution to problem 2 of HMM
void predict_state_sequence(){
	for (int t = 1; t <= T; t++)
	{
		long double max = 0;
		int index = 0;
		for (int j = 1; j <= N; j++)
		{
			if (gamma[t][j] > max)
			{
				max = gamma[t][j];
				index = j;
			}
		}
		q[t] = index;
	}
	FILE* fp=fopen("predicted_seq_gamma.txt","w");
	//printf("\nState Sequence\n");
	for (int t = 1; t <= T; t++)
	{
		fprintf(fp,"%4d\t",O[t]);
	}
	fprintf(fp,"\n");
	for (int t = 1; t <= T; t++)
	{
		//printf("%d ", q[t]);
		fprintf(fp,"%4d\t",q[t]);
	}
	fprintf(fp,"\n");
	fclose(fp);
	//printf("\n");
}

//Calculate Gamma
void calculate_gamma()
{
	for (int t = 1; t <= T; t++)
	{
		long double sum = 0;
		for (int i = 1; i <= N; i++)
		{
			sum += alpha[t][i] * beta[t][i];
		}
		for (int i = 1; i <= N; i++)
		{
			gamma[t][i] = alpha[t][i] * beta[t][i] / sum;
		}
	}
	FILE *fp=fopen("gamma.txt","w");
	//printf("Gamma Matrix\n");
	for (int t = 1; t <= T; t++)
	{
		for (int j = 1; j <= N; j++)
		{
			//printf("%.16e ", gamma[t][j]);
			fprintf(fp,"%.16e\t", gamma[t][j]);
		}
		fprintf(fp,"\n");
		//printf("\n");
	}
	fclose(fp);
	predict_state_sequence();
}

//Solution to Problem2 Of HMM
void viterbi_algo(){
	//Initialization
	for(int i=1;i<=N;i++){
		delta[1][i]=pi[i]*B[i][O[1]];
		psi[1][i]=0;
	}
	//Recursion
	for(int t=2;t<=T;t++){
		for(int j=1;j<=N;j++){
			long double max=DBL_MIN;
			int index=0;
			for(int i=1;i<=N;i++){
				if(delta[t-1][i]*A[i][j]>max){
					max=delta[t-1][i]*A[i][j];
					index=i;
				}
			}
			delta[t][j]=max*B[j][O[t]];
			psi[t][j]=index;
		}
	}
	//Termination
	P_star=DBL_MIN;
	for(int i=1;i<=N;i++){
		if(delta[T][i]>P_star){
			P_star=delta[T][i];
			q_star[T]=i;
		}
	}
	//State Sequence (Path) Backtracking
	for(int t=T-1;t>=1;t--){
		q_star[t]=psi[t+1][q_star[t+1]];
	}
	//print
	//printf("\nP*=%e\n",P_star);
	//printf("q* (state sequence):\n");
	FILE* fp=fopen("predicted_seq_viterbi.txt","w");
	for (int t = 1; t <= T; t++)
	{
		fprintf(fp,"%4d\t",O[t]);
	}
	fprintf(fp,"\n");
	for(int t=1;t<=T;t++){
		//printf("%d ",q_star[t]);
		fprintf(fp,"%4d\t",q_star[t]);
	}
	fprintf(fp,"\n");
	fclose(fp);
	//printf("\n");
}

//Calculate XI
void calculate_xi(){
	for(int t=1;t<T;t++){
		long double denominator=0.0;
		for(int i=1;i<=N;i++){
			for(int j=1;j<=N;j++){
				denominator+=(alpha[t][i]*A[i][j]*B[j][O[t+1]]*beta[t+1][j]);
			}
		}
		for(int i=1;i<=N;i++){
			for(int j=1;j<=N;j++){
				xi[t][i][j]=(alpha[t][i]*A[i][j]*B[j][O[t+1]]*beta[t+1][j])/denominator;
			}
		}
	}
}

//Reestimation; Solution to problem3 of HMM
void re_estimation(){
	//calculate Pi_bar
	for(int i=1;i<=N;i++){
		pi_bar[i]=gamma[1][i];
	}
	//calculate aij_bar
	for(int i=1;i<=N;i++){
		int mi=0;
		long double max_value=DBL_MIN;
		long double adjust_sum=0;
		for(int j=1;j<=N;j++){
			long double numerator=0.0, denominator=0.0;
			for(int t=1;t<=T-1;t++){
				numerator+=xi[t][i][j];
				denominator+=gamma[t][i];
			}
			A_bar[i][j]=(numerator/denominator);
			if(A_bar[i][j]>max_value){
				max_value=A_bar[i][j];
				mi=j;
			}
			adjust_sum+=A_bar[i][j];
		}
		A_bar[i][mi]+=(1-adjust_sum);
	}
	//calculate bjk_bar
	for(int j=1;j<=N;j++){
		int mi=0;
		long double max_value=DBL_MIN;
		long double adjust_sum=0;
		for(int k=1;k<=M;k++){
			long double numerator=0.0, denominator=0.0;
			for(int t=1;t<=T;t++){
				//if(q_star[t]==j){
					if(O[t]==k){
						numerator+=gamma[t][j];
					}
					denominator+=gamma[t][j];
				//}
			}
			B_bar[j][k]=(numerator/denominator);
			if(B_bar[j][k]>max_value){
				max_value=B_bar[j][k];
				mi=k;
			}
			if(B_bar[j][k]<1.00e-030){
				B_bar[j][k]=1.00e-030;
				//adjust_sum+=B_bar[j][k];
			}
			adjust_sum+=B_bar[j][k];
		}
		//B_bar[j][mi]-=adjust_sum;
		B_bar[j][mi]+=(1-adjust_sum);
		//printf("maxB index:%d\nadjust_sum=%.16e\nB_bar[j][mi]=%.16e\n",mi,adjust_sum,B_bar[j][mi]);
	}
	
	//update Pi_bar
	for(int i=1;i<=N;i++){
		pi[i]=pi_bar[i];
	}
	//upadte aij_bar
	for(int i=1;i<=N;i++){
		for(int j=1;j<=N;j++){
			A[i][j]=A_bar[i][j];
		}
	}
	//update bjk_bar
	for(int j=1;j<=N;j++){
		for(int k=1;k<=M;k++){
			B[j][k]=B_bar[j][k];
		}
	}
}

//Set initial model for each didgit
void set_initial_model(){
	for(int d=0;d<=6;d++){
		char srcfnameA[40];
		_snprintf(srcfnameA,40,"initial/A_%d.txt",d);
		char srcfnameB[40];
		_snprintf(srcfnameB,40,"initial/B_%d.txt",d);
		char destfnameA[40];
		_snprintf(destfnameA,40,"initial_model/A_%d.txt",d);
		char destfnameB[40];
		_snprintf(destfnameB,40,"initial_model/B_%d.txt",d);
		char copyA[100];
		_snprintf(copyA,100,"copy /Y %s %s",srcfnameA,destfnameA);
		char copyB[100];
		_snprintf(copyB,100,"copy /Y %s %s",srcfnameB,destfnameB);
		system(copyA);
		system(copyB);
	}
	
}

//Store initial values of HMM model parameter into arrays
void initial_model(int d){
	FILE *fp;
	//printf("T=%d\n",T);
	initialization();
	char filenameA[40];
	_snprintf(filenameA,40,"initial_model/A_%d.txt",d);
	fp = fopen(filenameA, "r");
	if (fp == NULL)
	{
		printf("Error\n");
	}
	//printf("A\n");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			fscanf(fp, "%Lf ", &A[i][j]);
			//printf("%.16e ", A[i][j]);
		}
		//printf("\n");
	}
	fclose(fp);

	//printf("B\n");
	char filenameB[40];
	_snprintf(filenameB,40,"initial_model/B_%d.txt",d);
	fp = fopen(filenameB, "r");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= M; j++)
		{
			//B[i][j]=(1.0)/M;
			//fprintf(fp, "%Lf ", B[i][j]);
			fscanf(fp, "%Lf ", &B[i][j]);
			//printf("%e ", B[i][j]);
		}
		//printf("\n");
	}
	fclose(fp);

	//printf("PI\n");
	fp = fopen("initial_model/pi.txt", "r");
	for (int i = 1; i <= N; i++)
	{
		fscanf(fp, "%Lf ", &pi[i]);
		//printf("%.16e ", pi[i]);
	}
	//printf("\n");
	fclose(fp);

	fp=fopen("o.txt","r");
	//printf("O\n");
	for (int i = 1; i <= T; i++)
	{
		fscanf(fp, "%d\t", &O[i]);
		//printf("%d ", O[i]);
	}
	//printf("\n");
	fclose(fp);
}

//Train HMM Model for given digit and given utterance
void train_model(int digit, int utterance){
	int m=0;
	//T=85;
	do{
		calculate_alpha();
		calculate_beta();
		calculate_gamma();
		P_star_dash=P_star;
		viterbi_algo();
		calculate_xi();
		re_estimation();
		m++;
		printf("Training Digit : %d\tIteration : %d\t\tp* = %e\n",digit,m,P_star);
	}while(m<30 && P_star - P_star_dash>1e-18);
	
	//Store A in file
	FILE *fp;
	char filenameA[40];
	
	_snprintf(filenameA,40,"234101048_lambda/A_%d_%d.txt",digit,utterance);
	fp=fopen(filenameA,"w");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			fprintf(fp, "%e ", A[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);

	//Store B in file
	char filenameB[40];
	
	_snprintf(filenameB,40,"234101048_lambda/B_%d_%d.txt",digit,utterance);
	fp=fopen(filenameB,"w");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= M; j++)
		{
			fprintf(fp, "%e ", B[i][j]);
			//printf("%e ", B[i][j]);
		}
		//printf("\n");
		fprintf(fp,"\n");
	}
	fclose(fp);
}

//Calculate average model parameter for given digit
void calculate_avg_model_param(int d){
	long double A_sum[N+1][N+1]={0};
	long double B_sum[N+1][M+1]={0};
	long double temp;
	FILE* fp;
	for(int u=1;u<=20;u++){
		char filenameA[40];
		
		_snprintf(filenameA,40,"234101048_lambda/A_%d_%d.txt",d,u);
		fp=fopen(filenameA,"r");
		for (int i = 1; i <= N; i++)
		{
			for (int j = 1; j <= N; j++)
			{
				fscanf(fp, "%Lf ", &temp);
				A_sum[i][j]+=temp;
				
			}
			
		}
		fclose(fp);
		char filenameB[40];
		
		_snprintf(filenameB,40,"234101048_lambda/B_%d_%d.txt",d,u);
		fp=fopen(filenameB,"r");
		for (int i = 1; i <= N; i++)
		{
			for (int j = 1; j <= M; j++)
			{
				fscanf(fp, "%Lf ", &temp);
				B_sum[i][j]+=temp;
				
			}
		}
		fclose(fp);
	}
	FILE* avgfp;
	char fnameA[40];
	_snprintf(fnameA,40,"initial_model/A_%d.txt",d);
	avgfp=fopen(fnameA,"w");
	for(int i=1;i<=N;i++){
		for(int j=1;j<=N;j++){
			A[i][j]=A_sum[i][j]/20;
			fprintf(avgfp,"%e ", A[i][j]);
		}
		fprintf(avgfp,"\n");
	}
	fclose(avgfp);
	char fnameB[40];
	_snprintf(fnameB,40,"initial_model/B_%d.txt",d);
	avgfp=fopen(fnameB,"w");
	for(int i=1;i<=N;i++){
		for(int j=1;j<=M;j++){
			B[i][j]=B_sum[i][j]/20;
			fprintf(avgfp,"%e ", B[i][j]);
		}
		fprintf(avgfp,"\n");
	}
	fclose(avgfp);
}

//Store converged Model Parameter
void store_final_lambda(int digit){
	FILE *fp;
	char filenameA[40];

	_snprintf(filenameA,40,"234101048_lambda/A_%d.txt",digit);
	fp=fopen(filenameA,"w");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			fprintf(fp, "%e ", A[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
	char filenameB[40];
	
	_snprintf(filenameB,40,"234101048_lambda/B_%d.txt",digit);
	fp=fopen(filenameB,"w");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= M; j++)
		{
			fprintf(fp, "%e ", B[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}


//Store model parameters of given digit in array for test input
void processTestFile(int d){
	FILE *fp;
	initialization();
	char filenameA[40];
	
	_snprintf(filenameA,40,"234101048_lambda/A_%d.txt",d);
	fp=fopen(filenameA,"r");
	if (fp == NULL){
		printf("Error\n");
	}
	for (int i = 1; i <= N; i++){
		for (int j = 1; j <= N; j++){
			fscanf(fp, "%Lf ", &A[i][j]);
		}
	}
	fclose(fp);

	char filenameB[40];
	
	_snprintf(filenameB,40,"234101048_lambda/B_%d.txt",d);
	fp=fopen(filenameB,"r");
	for (int i = 1; i <= N; i++){
		for (int j = 1; j <= M; j++){
			fscanf(fp, "%Lf ", &B[i][j]);
		}
	}
	fclose(fp);

	fp = fopen("initial_model/pi.txt", "r");
	for (int i = 1; i <= N; i++)
	{
		fscanf(fp, "%Lf ", &pi[i]);
	}
	fclose(fp);

	fp=fopen("o.txt","r");
	for (int i = 1; i <= T; i++)
	{
		fscanf(fp, "%d\t", &O[i]);
	}
	fclose(fp);
}

//recognize digit as max probability of all digit models
int recognize_digit(){
	int rec_digit=10;
	long double max_prob=DBL_MIN;
	for(int d=0;d<=6;d++){
		processTestFile(d);
		calculate_alpha();
		long double prob=calculate_score();
		
		if(prob>max_prob){
			max_prob=prob;
			rec_digit=d;
		}

		

	}
	printf("P(O/Lambda %d) = %e\t\t",rec_digit,max_prob);
	return rec_digit;
}

//Train HMM for given dataset
void train_HMM(){
	//Initialize A,B,PI as Intertia Model
	set_initial_model();
	printf("\nTraining Model\n");
	printf ("-------------------------------------------------------------------------------------------------------------------------------------\n");
	for(int d=0;d<=6;d++){
		for(int t=1;t<=2;t++){
			for(int u=1;u<=TRAIN_SIZE;u++){
				char filename[40];
				_snprintf(filename,40,"234101048_dataset/234101048_E_%d_%d.txt",d,u);
				//printf(filename);
				generate_observation_sequence(filename);
				initial_model(d);
				train_model(d,u);
			}
			calculate_avg_model_param(d);
		}
		store_final_lambda(d);
	}
}

//Test HMM for given dataset
void test_HMM(){
	double accuracy=0.0;
	int correct=0;
	int wrong=0;
	int total =0;
	//printf("Recognition on Testing Data\n");
	//printf("File Name\t\tDigit\tHighest Probability\t\tRecognized Digit\tCorrect/Wrong\n");
	//printf ("-------------------------------------------------------------------------------------------------------------------------------------\n");
	for(int d=0;d<=6;d++){
		for(int u=21;u<=30;u++){
			char filename[40];
			
			_snprintf(filename,40,"234101048_dataset/234101048_E_%d_%d.txt",d,u);
			generate_observation_sequence(filename);
			//printf("234101048_E_%d_%d.txt\t%d\t",d,u,d);
			int rd=recognize_digit();
			//printf("%d\t\t",rd);
			if(rd==d){
				//printf("CORRECT\n");
				accuracy+=1.0;
				correct++;
				total++;
			}
			else
			{
				//printf("WRONG\n");
				wrong++;
				total++;
			}
		}
	}
	//accuracy/=TEST_SIZE;
	/*printf ("-------------------------------------------------------------------------------------------------------------------------------------\n");
	printf("Total Test Instances :%d\n",total);
	printf("Correct Predictions :%d\n",correct);
	printf("Wrong Predictions :%d\n",wrong); 
	printf("Accuracy:%f\n",accuracy/total*100);*/
}

//hardcoded live testing as middle part
void process_live_data(char filename[100]){
	FILE *fp;
	char prefixf[100]="live_input/";
	strcat(prefixf,filename);
	fp=fopen(prefixf,"r");
	int samples[13000];
	int x=0;
	for(int i=0;!feof(fp);i++){
		fscanf(fp,"%d",&x);
		if(i>=6000 && i<19000){
			samples[i-6000]=x;
		}
	}
	fclose(fp);
	char prefix[100]="live_input/processed_";
	strcat(prefix,filename);
	fp=fopen(prefix,"w");
	for(int i=0;i<13000;i++){
		fprintf(fp,"%d\n",samples[i]);
	}
	fclose(fp);
}

//Live testing of Digit Recognition HMM Model
int live_test_HMM() {
    // Placeholder function to simulate live testing of HMM digit recognition
    // In practice, this would involve recording and processing audio input
    Sleep(2000);
    system("Recording_Module.exe 2 live_input/test.wav live_input/test.txt");
    process_live_data("test.txt");
    generate_observation_sequence("live_input/test.txt");
    int rd = recognize_digit();
    printf("Recognized Digit:%d\n", rd);
	Sleep(2000);
    return rd;
}

int main()
{
	printf ("------------------------------------------------------- Speech Based Connect 4 ------------------------------------------------------\n");
	printf ("                                                                                                                     SAYAN PAL       \n");
	printf ("                                                                                                                     234101048       \n");
	//printf (" Creating Universe from Speech Samples using LPC ....\n");
	generate_universe();
	//printf (" Universe Created !\n");
	//printf ("-------------------------------------------------------------------------------------------------------------------------------------\n");
	//printf (" Generating Codebook from Universe using LBG Algorithm \n");
	generate_codebook();
	load_codebook();
	//printf ("-------------------------------------------------------------------------------------------------------------------------------------\n");
	//Train
	
	train_HMM();
	
	//printf ("-------------------------------------------------------------------------------------------------------------------------------------\n");

	//Test
	test_HMM();
	printf ("-------------------------------------------------------------------------------------------------------------------------------------\n");

	srand(time(0)); // Initialize random seed
    int board[ROWS][COLS];
    create_board(board);
    bool game_over = false;
    int turn = 0;

    while (!game_over) {
        if (turn == 0) {
            int col = live_test_HMM(); // Get predicted digit from HMM
            printf("Player 1 (HMM predicted) makes selection: %d\n", col);
            if (is_valid_location(board, col)) {
                int row = get_next_open_row(board, col);
                drop_piece(board, row, col, 1);
                if (winning_move(board, 1)) {
                    printf("Player 1 wins!\n");
                    game_over = true;
                }
                print_board(board);
                printf("\n");
            }
        } else {
            int result[2];
            minimax(board, 5, -2147483648, 2147483647, true, result);
            int col = result[0];
            if (is_valid_location(board, col)) {
                int row = get_next_open_row(board, col);
                drop_piece(board, row, col, 2);
                if (winning_move(board, 2)) {
                    printf("AI wins!\n");
                    game_over = true;
                }
                print_board(board);
            }
        }
        turn = 1 - turn;
    }
	getchar();
    return 0;
}



