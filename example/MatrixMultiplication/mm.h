#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
using namespace std;

#define SIZE 128
#define TILE_SIZE 8

int *MAT_A,*MAT_B;
int *outCPU,*outGPU_share,*outGPU_nonshare;

void init()
{
	int index,tmp;
	
	MAT_A= new int[SIZE*SIZE];
	MAT_B= new int[SIZE*SIZE];
	outCPU= new int[SIZE*SIZE];
	outGPU_share= new int[SIZE*SIZE];
	outGPU_nonshare= new int[SIZE*SIZE];
	ifstream ifs;
	
	
	ifs.open("./A-Matrix.txt", ifstream::in);
	if(!ifs.is_open()){
		cout << "Can not open the input file A\n";
	}
	
	for(int i=0 ; i<SIZE ;i++){
		for(int j=0 ; j<SIZE ;j++){	
			ifs >> tmp;
			index = i*SIZE+j;
			MAT_A[index] = tmp;
		}
	}		
	ifs.close();
	
	
	ifs.open("./B-Matrix.txt", ifstream::in);
	if(!ifs.is_open()){
		cout << "Can not open the input file B\n";
	}
	
	for(int i=0 ; i<SIZE ;i++){
		for(int j=0 ; j<SIZE ;j++){	
			ifs >> tmp;
			index = i*SIZE+j;
			MAT_B[index] = tmp;
		}
	}		
	ifs.close();
		
}

bool checker()
{
	for(int i=0;i<SIZE;i++){
		for(int j=0;j<SIZE;j++){
			if( (outCPU[i*SIZE+j]!=outGPU_share[i*SIZE+j]) || (outCPU[i*SIZE+j]!=outGPU_nonshare[i*SIZE+j]) || (outGPU_share[i*SIZE+j]!=outGPU_share[i*SIZE+j]) ){
				cout << "The element[" << i*SIZE+j << "]is wrong" << endl; 
				cout << outCPU[i*SIZE+j] << " " << outGPU_share[i*SIZE+j] << " " << outGPU_nonshare[i*SIZE+j] << endl;
				return false;
			}
		}
	}
	
	return true;
}

int timespec_diff_ns(timespec& t1, timespec& t2)
{                                                                                
  return (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec) ;        
} 

