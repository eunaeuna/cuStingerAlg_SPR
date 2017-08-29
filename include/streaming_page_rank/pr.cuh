#pragma once

#include "algs.cuh"

#define PR_UPDATE 1

#if PR_UPDATE
#include "update.hpp"
#endif

namespace cuStingerAlgs {

typedef float prType;
class pageRankUpdate{
public:
	prType* prevPR;
	prType* currPR;
	prType* absDiff;
	// void* reduction;
	prType* reductionOut;
	prType* contri;
 
	length_t iteration;
	length_t iterationMax;
	length_t nv;
	prType threshold;
	prType damp;
	prType normalizedDamp;
#if 1 //queue	
	prType epsilon; //determinant for enqueuing    
	vertexQueue queue1;
//	pairPropQueue queue2;
	vertexQueue queue2;
	vertexQueue queue3;
	vertexQueue queueDlt; //delta queue
	length_t* visited;
	length_t* visitedDlt;
	length_t* usedOld;
	prType* diffPR;
    prType* delta;

	vertexId_t* vArraySrc;
	vertexId_t* vArrayDst;
#endif
	
};

// Label propogation is based on the values from the previous iteration.
class StreamingPageRank:public StaticAlgorithm{
public:
	virtual void Init(cuStinger& custing);
	virtual void Reset();
	virtual void Run(cuStinger& custing);
	virtual void Release();
#if PR_UPDATE
	//StreamingPageRank();
	//~StreamingPageRank();
	int sumTest(int n_numbers, int* numbers);

	void recomputeInsertionContriUndirected(cuStinger* custing, vertexId_t src, vertexId_t dst, void* metadata);
	void recomputeDeletionContriUndirected(cuStinger* custing, vertexId_t src, vertexId_t dst, void* metadata);

	void UpdateInsertionDiff(cuStinger& custing, BatchUpdateData& bud, length_t* len);
	void UpdateDeletionDiff(cuStinger& custing, BatchUpdateData& bud, length_t* len);
	void Run2(cuStinger& custing);
#endif
	void SyncHostWithDevice(){
		copyArrayDeviceToHost(devicePRData,&hostPRData,1, sizeof(pageRankUpdate));
	}
	void SyncDeviceWithHost(){
		copyArrayHostToDevice(&hostPRData,devicePRData,1, sizeof(pageRankUpdate));
	}
	void setInputParameters(length_t iterationMax = 20, prType threshold = 0.000001 ,prType damp=0.85);
	
	length_t getIterationCount();

	// User is responsible for de-allocating memory.
	prType* getPageRankScoresHost(){
		prType* hostArr = (prType*)allocHostArray(hostPRData.nv, sizeof(prType));
		copyArrayDeviceToHost(hostPRData.currPR, hostArr, hostPRData.nv, sizeof(prType) );
		return hostArr;
	}

	// User sends pre-allocated array.
	void getPageRankScoresHost(vertexId_t* hostArr){
		copyArrayDeviceToHost(hostPRData.currPR, hostArr, hostPRData.nv, sizeof(prType) );
	}

	void printRankings(cuStinger& custing);

protected: 
	pageRankUpdate hostPRData, *devicePRData;
	length_t reductionBytes;
private: 
	cusLoadBalance* cusLB;	
};


class StreamingPageRankOperator{
public:
static __device__ void init(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;
	pr->absDiff[src]=pr->currPR[src]=0.0;
	pr->prevPR[src]=1/float(pr->nv);
	// printf("%f, ", pr->prevPR[src]);
	*(pr->reductionOut)=0;
}

static __device__ void resetCurr(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;
	pr->currPR[src]=0.0;
	*(pr->reductionOut)=0;
}

static __device__ void resetContribution(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;
	pr->contri[src]=0.0;
}

static __device__ void computeContribuitionPerVertex(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;
	length_t sizeSrc = custing->dVD->getUsed()[src];
	if(sizeSrc==0)
		pr->contri[src]=0.0;
	else
		pr->contri[src]=pr->prevPR[src]/sizeSrc;

    if(src == 368)
		printf("contri[368]= %e\n", pr->contri[src]);
}


static __device__ void addContribuitions(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;
	atomicAdd(pr->currPR+dst,pr->contri[src]);
}

static __device__ void addContribuitionsUndirected(cuStinger* custing,vertexId_t src, vertexId_t dst, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;
	atomicAdd(pr->currPR+src,pr->contri[dst]);

}

static __device__ void dampAndDiffAndCopy(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;
	// pr->currPR[src]=(1-pr->damp)/float(pr->nv)+pr->damp*pr->currPR[src];
//    if(src == 125 || src == 126)
//		printf("++++currPR[%d]= %e\n", src, pr->currPR[src]);

	pr->currPR[src]=pr->normalizedDamp+pr->damp*pr->currPR[src];

//    if(src == 125 || src == 126)
//		printf("====currPR[%d]= %e\n", src, pr->currPR[src]);

	pr->absDiff[src]= fabsf(pr->currPR[src]-pr->prevPR[src]);
	pr->prevPR[src]=pr->currPR[src];
}

static __device__ void sum(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;
	atomicAdd(pr->reductionOut,pr->absDiff[src] );
}

static __device__ void sumPr(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;
	atomicAdd(pr->reductionOut,pr->prevPR[src] );
}

//update
#define PR_UPDATE 1
#if PR_UPDATE 

static __device__ void clearVisited(cuStinger* custing,vertexId_t src, void* metadata){
        pageRankUpdate* pr = (pageRankUpdate*)metadata;
        pr->visited[src] = 0;
        //pr->visitedDlt[src] = 0;
}
static __device__ void clearVisitedDlt(cuStinger* custing,vertexId_t src, void* metadata){
        pageRankUpdate* pr = (pageRankUpdate*)metadata;
        pr->visitedDlt[src] = 0;
}
static __device__ void clearDelta(cuStinger* custing,vertexId_t src, void* metadata){
        pageRankUpdate* pr = (pageRankUpdate*)metadata;
        pr->delta[src] = 0.0;
}

static __device__ void clear(cuStinger* custing,vertexId_t src, void* metadata){
        pageRankUpdate* pr = (pageRankUpdate*)metadata;
        pr->delta[src] = 0.0;
        pr->contri[src] = 0.0;
        pr->diffPR[src] = 0.0;
        pr->visited[src] = 0;
        pr->visitedDlt[src] = 0;
}

static __device__ void printQueue1(cuStinger* custing, vertexId_t src, void* metadata){
		//printf("Q1: %d\n", src);
}

static __device__ void printQueue2(cuStinger* custing, vertexId_t src, void* metadata){
		//printf("Q2: %d\n", src);
}
static __device__ void printQueueD(cuStinger* custing, vertexId_t src, void* metadata){
		//printf("QD: %d\n", src);
}

//Q1
static __device__ void recomputeInsertionContriUndirected(cuStinger* custing, vertexId_t src, vertexId_t dst, void* metadata){
        pageRankUpdate* pr = (pageRankUpdate*)metadata;
        length_t sizeDst = custing->dVD->getUsed()[dst]; //new
        length_t sizeSrc = custing->dVD->getUsed()[src]; //new
        if (sizeDst == 0) return; // no meaning in undirected graph
        //prType updateDiff = pr->damp*(pr->prevPR[dst]/(sizeDst));
        prType updateDiff = pr->damp*(pr->prevPR[dst]/(pr->usedOld[dst])); //@@old dst
        prType updateProp = pr->damp*(updateDiff/sizeSrc);

        if(pr->usedOld[dst] != sizeDst)
        	printf("#edges[%d]: old = %d, new = %d, updateDiff = %e\n", 
        		dst, pr->usedOld[dst], sizeDst, updateDiff);

        //atomicAdd(pr->currPR+src,updateDiff);
        atomicAdd(pr->diffPR+src,updateDiff);
        atomicAdd(pr->contri+src,updateDiff); //$$pair with updateDeltaAndMove

        if(fabs(updateProp) > pr->epsilon){
        	//atomicAdd(pr->contri+src,updateDiff);
        	//atomicAdd(pr->diffPR+src,-updateDiff); //<--- contri of duplication
	        if (pr->visited[src] == 0) {
	        	//CAS: old == compare ? val : old
	        	length_t temp = pr->visited[src] + 1;
	        	length_t old = atomicCAS(pr->visited+src,0,temp);
	        	if (old == 0) { 
	        		pr->queue2.enqueue(src); 
	        	} 	
	        }
        }else{
        	atomicAdd(pr->delta+src,updateDiff);
	        if ((pr->visited[src] == 0) && (pr->visitedDlt[src] == 0 )) {
	        	//CAS: old == compare ? val : old
	        	length_t temp = pr->visitedDlt[src] + 1;
	        	length_t old = atomicCAS(pr->visitedDlt+src,0,temp);
	        	if (old == 0) { 
	        		pr->queueDlt.enqueue(src); 
	        	} 	
	        }
        }     
}

static __device__ void recomputeDeletionContriUndirected(cuStinger* custing, vertexId_t src, vertexId_t dst, void* metadata){
        pageRankUpdate* pr = (pageRankUpdate*)metadata;
        length_t sizeDst = custing->dVD->getUsed()[dst]; //new
        length_t sizeSrc = custing->dVD->getUsed()[src]; //new
        if (sizeDst == 0) return; // no meaning in undirected graph
        //prType updateDiff = pr->damp*(pr->prevPR[dst]/(sizeDst));
        prType updateDiff = pr->damp*(pr->prevPR[dst]/(pr->usedOld[dst])); //@@old dst
        prType updateProp = pr->damp*(updateDiff/sizeSrc);

        //if(pr->usedOld[dst] != sizeDst)
        //if(dst == 368)
        //	printf("#edges[%d]: old = %d, new = %d, diff = %e, diff = %e\n", 
        //		dst, pr->usedOld[dst], sizeDst, updateDiff, updateDiff);

        //atomicAdd(pr->currPR+src,updateDiff);
        atomicAdd(pr->diffPR+src,updateDiff);
        atomicAdd(pr->contri+src,updateDiff); //$$pair with updateDeltaAndMove

        if(fabs(updateProp) > pr->epsilon){
        	//atomicAdd(pr->contri+src,updateDiff);
        	//atomicAdd(pr->diffPR+src,-updateDiff); //<--- contri of duplication
	        if (pr->visited[src] == 0) {
	        	//CAS: old == compare ? val : old
	        	length_t temp = pr->visited[src] + 1;
	        	length_t old = atomicCAS(pr->visited+src,0,temp);
	        	if (old == 0) { 
	        		pr->queue2.enqueue(src); 
	        	} 	
	        }
        }else{
        	atomicAdd(pr->delta+src,updateDiff);
	        if ((pr->visited[src] == 0) && (pr->visitedDlt[src] == 0 )) {
	        	//CAS: old == compare ? val : old
	        	length_t temp = pr->visitedDlt[src] + 1;
	        	length_t old = atomicCAS(pr->visitedDlt+src,0,temp);
	        	if (old == 0) { 
	        		pr->queueDlt.enqueue(src); 
	        	} 	
	        }
        }     
}

//Q2
static __device__ void updateContributionsUndirected(cuStinger* custing, vertexId_t src, vertexId_t dst, void* metadata){

	    pageRankUpdate* pr = (pageRankUpdate*)metadata;
        length_t sizeSrc = custing->dVD->getUsed()[src];
        length_t sizeDst = custing->dVD->getUsed()[dst];

        if(sizeSrc == 0) return; //no meaning in undirected graph

        //prType updateDiff = pr->damp*((pr->currPR[src]/(sizeSrc))-(pr->prevPR[src]/(sizeSrc))); //
        prType updateDiff = pr->damp*((pr->currPR[src]/(sizeSrc))-(pr->prevPR[src]/(pr->usedOld[src]))); //-1 <----usedOld
        prType updateProp = pr->damp*(updateDiff/sizeDst);

        //if(pr->usedOld[dst] != sizeDst)
        //	printf("#edges[%d]: old = %d, new = %d\n", dst, pr->usedOld[dst], sizeDst);

		//atomicAdd(pr->contri+dst,updateProp);
		atomicAdd(pr->contri+dst,updateDiff); //$$pair with updateDeltaAndMove

        if(dst == 368)
        	printf("#edges[%d -> %d]: old = %d, new = %d, diff = %e, contri = %e\n", 
        		src, dst, pr->usedOld[dst], sizeDst, updateDiff, pr->contri[dst]);

        if(fabs(updateProp) > pr->epsilon){
        	//atomicAdd(pr->currPR+dst,updateDiff);
        	//atomicAdd(pr->contri+dst,updateDiff);
	        if (pr->visited[dst] == 0) {
	        	//CAS: old == compare ? val : old
	        	length_t temp = pr->visited[dst] + 1;
	        	length_t old = atomicCAS(pr->visited+dst,0,temp);
	        	if (old == 0) {
	        		//printf("updateStepQ[\t%d,\t%d]:\t, e=\t%e, Diff=\t%e, Prop=\t%e, currPR[%d]=\t%e, prevPR[%d]=\t%e, sizeDst=\t%d \n", src, dst, pr->epsilon, updateDiff, updateProp, dst, pr->currPR[dst], dst, pr->prevPR[dst], sizeDst);
	        		pr->queue2.enqueue(dst); 
	        	} 	
	        }
        }else{
        	atomicAdd(pr->delta+dst,updateDiff);
        	//atomicAdd(pr->delta+dst,updateProp);
	        if ((pr->visited[dst] == 0) && (pr->visitedDlt[dst] == 0)) {
	        	//CAS: old == compare ? val : old
	        	length_t temp = pr->visitedDlt[dst] + 1;
	        	length_t old = atomicCAS(pr->visitedDlt+dst,0,temp);
	        	if (old == 0) { 
	        		//printf("updateStepD[\t%d,\t%d]:\t, e=\t%e, Diff=\t%e, Prop=\t%e, currPR[%d]=\t%e, prevPR[%d]=\t%e, sizeDst=\t%d \n", src, dst, pr->epsilon, updateDiff, updateProp, dst, pr->currPR[dst], dst, pr->prevPR[dst], sizeDst);
	        		pr->queueDlt.enqueue(dst); 
	        	} 	
	        }
        }
}
#endif

static __device__ void removeContributionsUndirected(cuStinger* custing, vertexId_t src, void* metadata){
	    pageRankUpdate* pr = (pageRankUpdate*)metadata;
	    prType diffPR = pr->diffPR[src];
		atomicAdd(pr->currPR+src,-diffPR);
		//atomicAdd(pr->currPR+src,-(pr->diffPR[src]));
}

static __device__ void updateDeltaAndMove(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;
	if(pr->delta[src] > pr->epsilon)
	{
		if (pr->visited[src] == 0) {
	       	//CAS: old == compare ? val : old
	       	length_t temp = pr->visited[src] + 1;
	       	length_t old = atomicCAS(pr->visited+src,0,temp);
	        if (old == 0) {
	        	//prType delta = pr->delta[src]; //$$pair with recomputeContributionUndirected, updateContributionsUndirected
	        	//atomicAdd(pr->contri+src,delta);
	        	pr->delta[src] = 0.0;
	        	pr->queue2.enqueue(src); 
	        } 	
	    }
	}
}

static __device__ void updateDeltaAndCopy(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;

    prType delta = pr->delta[src];
    atomicAdd(pr->currPR+src,delta);
	//atomicAdd(pr->currPR+src,pr->delta[src]);
}

static __device__ void updateContributionAndCopy(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;

    prType contri = pr->contri[src];
    atomicAdd(pr->currPR+src,contri);
	//atomicAdd(pr->currPR+src,pr->contri[src]);
}

static __device__ void updateContributionsDirected(cuStinger* custing, vertexId_t src, vertexId_t dst, void* metadata){

}

static __device__ void updateDiffAndCopy(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;

	pr->absDiff[src]= fabsf(pr->currPR[src]-pr->prevPR[src]); //adsDiff --> delta[nv]
	pr->prevPR[src]=pr->currPR[src];
}

static __device__ void updateSum(cuStinger* custing,vertexId_t src, void* metadata){
	pageRankUpdate* pr = (pageRankUpdate*)metadata;
	atomicAdd(pr->reductionOut,pr->absDiff[src] );
}


static __device__ void setIds(cuStinger* custing,vertexId_t src, void* metadata){
	vertexId_t* ids = (vertexId_t*)metadata;
	ids[src]=src;
}

static __device__ void print(cuStinger* custing,vertexId_t src, void* metadata){
	int* ids = (int*)metadata;
	if(threadIdx.x==0 & blockIdx.x==0){
		// printf("The wheels on the bus go round and round and round and round %d\n",*ids);
	}
}


// static __device__ void addDampening(cuStinger* custing,vertexId_t src, void* metadata){
// 	pageRankUpdate* pr = (pageRankUpdate*)metadata;
// 	pr->currPR[src]=(1-pr->damp)/float(pr->nv)+pr->damp*pr->currPR[src];
// }

// static __device__ void absDiff(cuStinger* custing,vertexId_t src, void* metadata){
// 	pageRankUpdate* pr = (pageRankUpdate*)metadata;
// 	pr->absDiff[src]= abs(pr->currPR[src]-pr->prevPR[src]);
// }

 static __device__ void prevEqualCurr(cuStinger* custing,vertexId_t src, void* metadata){
 	pageRankUpdate* pr = (pageRankUpdate*)metadata;
 	pr->prevPR[src]=pr->currPR[src];
}



};

} // cuStingerAlgs namespace
