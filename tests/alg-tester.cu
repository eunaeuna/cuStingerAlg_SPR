#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <math.h>


#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

#include "algs.cuh"

#include "static_breadth_first_search/bfs_top_down.cuh"
#include "static_breadth_first_search/bfs_bottom_up.cuh"
#include "static_breadth_first_search/bfs_hybrid.cuh"
#include "static_connected_components/cc.cuh"
#include "static_page_rank/pr.cuh"
#include "static_betweenness_centrality/bc.cuh"

#include "streaming_page_rank/pr.cuh"

using namespace cuStingerAlgs;


#define CUDA(call, ...) do {                        \
        cudaError_t _e = (call);                    \
        if (_e == cudaSuccess) break;               \
        fprintf(stdout,                             \
                "CUDA runtime error: %s (%d)\n",    \
                cudaGetErrorString(_e), _e);        \
        return -1;                                  \
    } while (0)

#define PR_UPDATE 1

#if PR_UPDATE
// RNG using Lehmer's Algorithm ================================================
#define RNG_A 16807
#define RNG_M 2147483647
#define RNG_Q 127773
#define RNG_R 2836
#define RNG_SCALE (1.0 / RNG_M)

// Seed can always be changed manually
static int seed = 1;
double getRand(){

    int k = seed / RNG_Q;
    seed = RNG_A * (seed - k * RNG_Q) - k * RNG_R;

    if (seed < 0) {
        seed += RNG_M;
    }

    return seed * (double) RNG_SCALE;
}

void generateEdgeUpdates(length_t nv, length_t numEdges, vertexId_t* edgeSrc, vertexId_t* edgeDst){
        printf("-----------------------------------------------\n");
        for(int32_t e=0; e<numEdges*2;e++)
        {
                edgeSrc[e] = 10*(e+1);//rand()%nv;
                edgeDst[e] = 20*(e+1);//rand()%nv;
                printf("edgeSrc[%d]=%d,\t edgeDst[%d]=%d\n",e,edgeSrc[e],e,edgeDst[e]);
                e++;
                edgeSrc[e] = edgeDst[e-1];
                edgeDst[e] = edgeSrc[e-1];
                printf("edgeSrc[%d]=%d,\t edgeDst[%d]=%d\n",e,edgeSrc[e],e,edgeDst[e]);
        }
}
#if 0
void generateEdgeUpdatesRMAT(length_t nv, length_t numEdges, vertexId_t* edgeSrc, vertexId_t* edgeDst,double A, double B, double C, double D){
        int64_t src,dst;
        int scale = (int)log2(double(nv));
        for(int32_t e=0; e<numEdges; e++){
                rmat_edge(&src,&dst,scale, A,B,C,D);
                edgeSrc[e] = src;
                edgeDst[e] = dst;
        }
}
#endif

void printcuStingerUtility(cuStinger custing, bool allInfo){
        length_t used,allocated;

        used     =custing.getNumberEdgesUsed();
        allocated=custing.getNumberEdgesAllocated();
        if (allInfo)
                cout << "getNumberEdgesUsed, " << used << ", " << allocated << ", " << (float)used/(float)allocated;
        else
                cout << "getNumberEdgesUsed, " << (float)used/(float)allocated;
}

#endif


int main(const int argc, char *argv[]){
#if 1
	int device=0;
    cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
 
    length_t nv, ne,*off;
    vertexId_t *adj;

	bool isDimacs,isSNAP,isRmat=false,isMarket;
	string filename(argv[1]);
	isDimacs = filename.find(".graph")==std::string::npos?false:true;
	isSNAP   = filename.find(".txt")==std::string::npos?false:true;
	isRmat 	 = filename.find("kron")==std::string::npos?false:true;
	isMarket = filename.find(".mtx")==std::string::npos?false:true;

	if(isDimacs){
	    readGraphDIMACS(argv[1],&off,&adj,&nv,&ne,isRmat);
	}
	else if(isSNAP){
	    readGraphSNAP(argv[1],&off,&adj,&nv,&ne,isRmat);
	}
	else if(isMarket){
		readGraphMatrixMarket(argv[1],&off,&adj,&nv,&ne,(isRmat)?false:true);
	}
	else{ 
		cout << "Unknown graph type" << endl;
	}

	cout << "Vertices: " << nv << "    Edges: " << ne << endl;

	cudaEvent_t ce_start,ce_stop;
	cuStinger custing(defaultInitAllocater,defaultUpdateAllocater);

	cuStingerInitConfig cuInit;
	cuInit.initState =eInitStateCSR;
	cuInit.maxNV = nv+1;
	cuInit.useVWeight = false;
	cuInit.isSemantic = false;  // Use edge types and vertex types
	cuInit.useEWeight = false;
	// CSR data
	cuInit.csrNV 			= nv;
	cuInit.csrNE	   		= ne;
	cuInit.csrOff 			= off;
	cuInit.csrAdj 			= adj;
	cuInit.csrVW 			= NULL;
	cuInit.csrEW			= NULL;

	custing.initializeCuStinger(cuInit);
	
	float totalTime;

#if 0
	StaticPageRank pr;
	pr.Init(custing);
	pr.Reset();
	pr.setInputParameters(50,0.000001);
	start_clock(ce_start, ce_stop);
	pr.Run(custing);
	totalTime = end_clock(ce_start, ce_stop);
	cout << "The number of iterations      : " << pr.getIterationCount() << endl;
	cout << "Total time for pagerank       : " << totalTime << endl; 
	cout << "Average time per iteartion "
			"   : " << totalTime/(float)pr.getIterationCount() << endl; 
	pr.printRankings(custing);

	pr.Release();
#endif

    StreamingPageRank upr;
    upr.Init(custing);
    upr.Reset();
    upr.setInputParameters(50,0.000001);

    start_clock(ce_start, ce_stop);
    upr.Run(custing);
    totalTime = end_clock(ce_start, ce_stop);
    cout << "=============================================" << endl;
    cout << "The number of iterations      : " << upr.getIterationCount() << endl;
    cout << "Total time for streaming pagerank       : " << totalTime << endl;
    cout << "Average time per iteartion    : " << totalTime/(float)upr.getIterationCount() << endl;
    upr.printRankings(custing);
    
#if 1

	//------------------------
    // store num of edges (usedOld[nv])
	// update
	//------------------------
	        //TO DO: adding termination conditions; iterations and epsilon
	        //upr.setInputParameters(30,0.0000001);  //initialize hostRPData

	        //graph update
	        length_t *len = (length_t *)malloc(sizeof(length_t)*(nv));
	        for(unsigned int i=0; i<nv; ++i){
	           len[i] = off[i+1] - off[i];
	           //printf("len[%d]: %d\n", i, len[i]); //oldUsed[i]
	        }

	        unsigned int numBatches = 1;
	        std::vector<BatchUpdateData*>buds(numBatches);
	        unsigned int numEdges = 10; //test
	        length_t numTotalEdges = numEdges;

	        for(unsigned int i=0; i<numBatches; ++i){
	            buds[i] = new BatchUpdateData(numTotalEdges*2,true,nv); //undirected
	        	//buds[i] = new BatchUpdateData(numTotalEdges,true,nv); //directed
	        }

	        for(unsigned int i=0; i<numBatches; ++i){
	           BatchUpdateData& bud = *buds[i];
               //vertexId_t *src = budi.getSrc();
               //vertexId_t *dst = budi.getDst();

	            if(isRmat){
	            //  double a = 0.55, b = 0.15, c = 0.15,d = 0.25;
	            //  generateEdgeUpdatesRMAT(nv, numEdges, bud.getSrc(),bud.getDst(),a,b,c,d);
	            }
	            else{
	                generateEdgeUpdates(nv, numEdges, bud.getSrc(),bud.getDst()); //undirected
            	//    generateEdgeUpdates(nv, numEdges, bud.getSrc(),bud.getDst()); //directed
	            }
	        }
#if 0	        
            // Add duplicates from beginning of batch
            for(unsigned i = 0; i < dupEgdes; ++i) {
                    src[numEdgesL + i] = src[i];
                    dst[numEdgesL + i] = dst[i];
            }
#endif

	        length_t *newOff = (length_t *)malloc(sizeof(length_t)*(nv+1));
	        length_t sum = 0;
	        for(unsigned int i=0; i<nv+1; ++i){
	           newOff[i] = sum;
	           sum += len[i];
	        }
	        vertexId_t *newAdj = (vertexId_t*)malloc(sizeof(vertexId_t)*(newOff[nv]));
	        //populate newAdj
	        for(unsigned int i=0, j=0; i<ne; ++i) {
	           if(adj[i] != -1) newAdj[j++] = adj[i];
	        }

	        cuInit.csrNE = newOff[nv];
	        cuInit.csrOff = newOff;
	        cuInit.csrAdj = newAdj;

	        cuStinger custingTest(defaultInitAllocater,defaultUpdateAllocater);
	        custingTest.initializeCuStinger(cuInit);

	        unsigned int sps = 128; //block size
	        length_t allocs;
	        BatchUpdate bu1(*buds[0]);
	        bu1.sortDeviceBUD(sps);
	        custingTest.edgeInsertionsSorted(bu1, allocs);

	        printcuStingerUtility(custingTest, true);//false);

	        start_clock(ce_start, ce_stop);

#define SPR_ON 1
#if SPR_ON //streaming pr
	        printf("\n<spr>======================================\n");
	        upr.UpdateInsertionDiff(custingTest, *buds[0], len); //updated graph
	        upr.setInputParameters(0,0.000001); //iterationMax, threshold, epsilon, damp
	        upr.Run2(custingTest);
#else
	        printf("\n<pr>======================================\n");
	        upr.setInputParameters(52,0.000001);
	        upr.Run(custingTest);
#endif	        
	        totalTime = end_clock(ce_start, ce_stop);	        
	        cout << "The number of iterations      : " << upr.getIterationCount() << endl;
	        cout << "Total time for updating streaming pagerank       : " << totalTime << endl;
	        cout << "Average time per iteartion    : " << totalTime/(float)upr.getIterationCount() << endl;
        
	        upr.printRankings(custingTest);
#define DEL_ON 0
#if DEL_ON
	        start_clock(ce_start, ce_stop);
	        printf("\n<deletion>======================================\n");
	        upr.UpdateDeletionDiff(custingTest, *buds[0], len); //updated graph
	        upr.setInputParameters(50,0.000001); //iterationMax, threshold, epsilon, damp
	        upr.Run2(custingTest);

	        totalTime = end_clock(ce_start, ce_stop);	        
	        cout << "The number of iterations      : " << upr.getIterationCount() << endl;
	        cout << "Total time for updating streaming pagerank       : " << totalTime << endl;
	        cout << "Average time per iteartion    : " << totalTime/(float)upr.getIterationCount() << endl;
        
	        upr.printRankings(custing);	        
#endif

#endif //end of update   

	custing.freecuStinger();

	free(off);
	free(adj);
    return 0;	
#endif
}

