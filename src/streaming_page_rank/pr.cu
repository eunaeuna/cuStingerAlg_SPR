#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <iomanip> 

#include <cub.cuh>
#include <util_allocator.cuh>

#include <device/device_reduce.cuh>
#include <kernel_mergesort.hxx>

#include "update.hpp"
#include "cuStinger.hpp"

#include "operators.cuh"
#include "streaming_page_rank/pr.cuh"

#define PR_UPDATE 1

using namespace cub;
using namespace mgpu;

CachingDeviceAllocator  g_u_allocator(true);  // Caching allocator for device memory

namespace cuStingerAlgs {  


//#ifndef LIB_H
//#define LIB_H

extern "C" {
    StreamingPageRank* StreamingPageRank_new() { return new StreamingPageRank(); }
    int StreamingPageRank_sumTest(StreamingPageRank* stpr, int n_num, int* numbers) { return stpr->sumTest(n_num, numbers); }
}
//#endif

int StreamingPageRank::sumTest(int n_numbers, int *numbers){
    int i;
    int sum;
    for (i = 0; i < n_numbers; i++) {
        sum += numbers[i];
    }
    return sum;
}

void StreamingPageRank::Init(cuStinger& custing){
	hostPRData.nv = custing.nv;
	hostPRData.prevPR  = (prType*) allocDeviceArray(hostPRData.nv+1, sizeof(prType));
	hostPRData.currPR  = (prType*) allocDeviceArray(hostPRData.nv+1, sizeof(prType));
	hostPRData.absDiff = (prType*) allocDeviceArray(hostPRData.nv+1, sizeof(prType));
	hostPRData.contri = (prType*) allocDeviceArray(hostPRData.nv+1, sizeof(prType));

	hostPRData.reductionOut = (prType*) allocDeviceArray(1, sizeof(prType));
	// hostPRData.reduction=NULL;

	devicePRData = (pageRankUpdate*)allocDeviceArray(1, sizeof(pageRankUpdate));
	hostPRData.queue1.Init(custing.nv);
	hostPRData.queue2.Init(custing.nv);
	hostPRData.queue3.Init(custing.nv);
	hostPRData.queueDlt.Init(custing.nv);
	hostPRData.visited = (length_t*)allocDeviceArray(hostPRData.nv+1, sizeof(length_t));
    hostPRData.visitedDlt = (length_t*)allocDeviceArray(hostPRData.nv+1, sizeof(length_t));
	hostPRData.usedOld = (length_t*)allocDeviceArray(hostPRData.nv+1, sizeof(length_t));
	hostPRData.diffPR = (prType*) allocDeviceArray(hostPRData.nv+1, sizeof(prType));
	hostPRData.delta = (prType*) allocDeviceArray(hostPRData.nv+1, sizeof(prType));

	SyncDeviceWithHost();

	//cusLB = new cusLoadBalance(custing.nv); //ERROR!!
	cusLB = new cusLoadBalance(custing);

	Reset();
}

void StreamingPageRank::Reset(){
	hostPRData.iteration = 0;
	hostPRData.queue1.resetQueue();
	hostPRData.queue2.resetQueue();
	hostPRData.queue3.resetQueue();
	hostPRData.queueDlt.resetQueue();
	SyncDeviceWithHost();
}

void StreamingPageRank::Release(){
	free(cusLB);	
	freeDeviceArray(devicePRData);
	freeDeviceArray(hostPRData.currPR);
	freeDeviceArray(hostPRData.prevPR);
	freeDeviceArray(hostPRData.absDiff);
	// freeDeviceArray(hostPRData.reduction);
	freeDeviceArray(hostPRData.reductionOut);
	freeDeviceArray(hostPRData.contri);
	
    //queue
    //hostPRData.queue1.Release();
    //hostPRData.queue2.Release();
    //hostPRData.queue3.Release();
    //hostPRData.queueDlt.Release();
    freeDeviceArray(hostPRData.visited);
    freeDeviceArray(hostPRData.visitedDlt);
    freeDeviceArray(hostPRData.usedOld);
    freeDeviceArray(hostPRData.delta);
    freeDeviceArray(hostPRData.diffPR);
}

void StreamingPageRank::Run(cuStinger& custing){

	allVinG_TraverseVertices<StreamingPageRankOperator::init>(custing,devicePRData);
	hostPRData.iteration = 0;

	prType h_out = hostPRData.threshold+1;

	while(hostPRData.iteration < hostPRData.iterationMax && h_out>hostPRData.threshold){
    //while(h_out>hostPRData.threshold){
		SyncDeviceWithHost();

		allVinA_TraverseVertices<StreamingPageRankOperator::resetCurr>(custing,devicePRData,*cusLB);
		allVinA_TraverseVertices<StreamingPageRankOperator::computeContribuitionPerVertex>(custing,devicePRData,*cusLB);
		allVinA_TraverseEdges_LB<StreamingPageRankOperator::addContribuitionsUndirected>(custing,devicePRData,*cusLB);
		allVinA_TraverseVertices<StreamingPageRankOperator::dampAndDiffAndCopy>(custing,devicePRData,*cusLB);

		allVinG_TraverseVertices<StreamingPageRankOperator::sum>(custing,devicePRData);
		SyncHostWithDevice();

		copyArrayDeviceToHost(hostPRData.reductionOut,&h_out, 1, sizeof(prType));
		h_out=hostPRData.threshold+1;
		//cout << "The number of elements : " << hostPRData.nv << endl;

		hostPRData.iteration++;

        //cout << "pr it: " << hostPRData.iteration << endl;
        //if(hostPRData.iteration > 1000) break;
	}
}

#if PR_UPDATE
void StreamingPageRank::Run2(cuStinger& custing){
    hostPRData.iteration = 0;
    prType h_out = hostPRData.threshold+1;

#if 1 //keep this because of contri used during update. need to recompute contribution of all vertices
    while(hostPRData.iteration < hostPRData.iterationMax && h_out>hostPRData.threshold){
        SyncDeviceWithHost();

        allVinA_TraverseVertices<StreamingPageRankOperator::resetCurr>(custing,devicePRData,*cusLB);
        allVinA_TraverseVertices<StreamingPageRankOperator::computeContribuitionPerVertex>(custing,devicePRData,*cusLB);
        allVinA_TraverseEdges_LB<StreamingPageRankOperator::addContribuitionsUndirected>(custing,devicePRData,*cusLB);
        allVinA_TraverseVertices<StreamingPageRankOperator::dampAndDiffAndCopy>(custing,devicePRData,*cusLB);

        allVinG_TraverseVertices<StreamingPageRankOperator::sum>(custing,devicePRData);
        SyncHostWithDevice();

        copyArrayDeviceToHost(hostPRData.reductionOut,&h_out, 1, sizeof(prType));
        h_out=hostPRData.threshold+1;
        //cout << "The number of elements : " << hostPRData.nv << endl;
        cout << "iteration: " << hostPRData.iteration << endl;

        hostPRData.iteration++;
    }
#else
    while(hostPRData.iteration < hostPRData.iterationMax && h_out>hostPRData.threshold){
        SyncDeviceWithHost();
#if 1
        allVinA_TraverseVertices<StreamingPageRankOperator::resetCurr>(custing,devicePRData,*cusLB);
#else
        allVinA_TraverseVertices<StreamingPageRankOperator::resetCurr>(custing,devicePRData,
                        hostPRData.queue2,hostPRData.queue2.getActiveQueueSize());        
#endif
        hostPRData.queue2.setQueueCurr(0); 
        length_t prevEnd = hostPRData.queue2.getQueueEnd();
        //printf("\n(%d /////////// currPR ---> prevPR \n", hostPRData.queue2.getActiveQueueSize());   

        allVinA_TraverseVertices<StreamingPageRankOperator::computeContribuitionPerVertex>(custing,devicePRData,*cusLB);
        //allVinA_TraverseEdges_LB<StreamingPageRankOperator::addContribuitionsUndirected>(custing,devicePRData,
        //                *cusLB,hostPRData.queue2,hostPRData.queue2.getActiveQueueSize());//batchsize        
        allVinA_TraverseEdges_LB<StreamingPageRankOperator::addContribuitionsUndirected>(custing,devicePRData,*cusLB);
        
        //allVinA_TraverseVertices<StreamingPageRankOperator::dampAndDiffAndCopy>(custing,devicePRData,
        //                hostPRData.queue2,hostPRData.queue2.getActiveQueueSize());
        allVinA_TraverseVertices<StreamingPageRankOperator::dampAndDiffAndCopy>(custing,devicePRData,*cusLB);

        allVinG_TraverseVertices<StreamingPageRankOperator::sum>(custing,devicePRData);
        SyncHostWithDevice();

        copyArrayDeviceToHost(hostPRData.reductionOut,&h_out, 1, sizeof(prType));
        h_out=hostPRData.threshold+1;
        //cout << "The number of elements : " << hostPRData.nv << endl;
        //cout << "iteration: " << hostPRData.iteration << endl;

        hostPRData.iteration++;
    }    
#endif    
}

void StreamingPageRank::UpdateInsertionDiff(cuStinger& custing, BatchUpdateData& bud, length_t* len) {

	length_t batchsize = *(bud.getBatchSize());
    vertexId_t *edgeSrc = bud.getSrc();
    vertexId_t *edgeDst = bud.getDst();
        
	hostPRData.vArraySrc = (vertexId_t*)allocDeviceArray(hostPRData.nv, sizeof(vertexId_t));
	hostPRData.vArrayDst = (vertexId_t*)allocDeviceArray(hostPRData.nv, sizeof(vertexId_t));
    
    copyArrayHostToDevice(edgeSrc,hostPRData.vArraySrc,batchsize,sizeof(vertexId_t));
    copyArrayHostToDevice(edgeDst,hostPRData.vArrayDst,batchsize,sizeof(vertexId_t));


    //insert vertices in updating list in Q1
    for(length_t i=0; i<batchsize; i++) {
      	hostPRData.queue1.enqueueFromHost(edgeSrc[i]);
        //cout << "queueu1 enqueue [" << i << "]: " << edgeSrc[i] << endl;
    }    

    copyArrayHostToDevice((length_t*)len,hostPRData.usedOld,hostPRData.nv,sizeof(length_t));

//---------------------------------------------------------------------------------------------------
    printf("\n--------------- update contri \n");
    SyncDeviceWithHost(); //added for threashold and iteration count
    allVinG_TraverseVertices<StreamingPageRankOperator::clear>(custing,devicePRData); //added

//---------------------------------------------------------------------------------------------------  
    printf("--------------- recompute \n");       
    allVinA_TraverseOneEdge<StreamingPageRankOperator::recomputeInsertionContriUndirected>(custing,devicePRData,
            		hostPRData.vArraySrc,hostPRData.vArrayDst,hostPRData.queue1.getActiveQueueSize());//batchsize*2);
    SyncHostWithDevice(); //copy Q2 //copyArrayDevicetoHost @@

    allVinA_TraverseVertices<StreamingPageRankOperator::updateDeltaAndMove>(custing,devicePRData,
                hostPRData.queueDlt,hostPRData.queueDlt.getActiveQueueSize());    
    SyncHostWithDevice();
    allVinA_TraverseVertices<StreamingPageRankOperator::updateContributionAndCopy>(custing,devicePRData,
                hostPRData.queue2,hostPRData.queue2.getActiveQueueSize());   
   
    length_t prevEnd = hostPRData.queue2.getQueueEnd(); //1 <-- Q1
    SyncDeviceWithHost();

    hostPRData.queueDlt.resetQueue();//$$$$$
    allVinA_TraverseVertices<StreamingPageRankOperator::clearVisitedDlt>(custing,devicePRData,
                hostPRData.queueDlt,hostPRData.queueDlt.getActiveQueueSize());
    allVinA_TraverseVertices<StreamingPageRankOperator::clearDelta>(custing,devicePRData,
                hostPRData.queueDlt,hostPRData.queueDlt.getActiveQueueSize());    
    SyncDeviceWithHost();

#if 1
    printf("--------------- propagate \n");       
    int i = 0;
    while(hostPRData.queue2,hostPRData.queue2.getActiveQueueSize()>0)
    { 
        printf("++Q2 size (%d): %d\n", i, hostPRData.queue2.getActiveQueueSize());

        allVinA_TraverseEdges_LB<StreamingPageRankOperator::updateContributionsUndirected>(custing,devicePRData,
                    *cusLB,hostPRData.queue2,hostPRData.queue2.getActiveQueueSize());//batchsize
        SyncHostWithDevice(); 
        hostPRData.queue2.setQueueCurr(prevEnd); 
        prevEnd = hostPRData.queue2.getQueueEnd(); //3 <-- 2st step
        SyncDeviceWithHost();
        allVinA_TraverseVertices<StreamingPageRankOperator::updateDeltaAndMove>(custing,devicePRData,
                    hostPRData.queueDlt,hostPRData.queueDlt.getActiveQueueSize());
        allVinA_TraverseVertices<StreamingPageRankOperator::updateContributionAndCopy>(custing,devicePRData,
                    hostPRData.queue2,hostPRData.queue2.getActiveQueueSize());

        if(i == 0){
            allVinA_TraverseVertices<StreamingPageRankOperator::removeContributionsUndirected>(custing,devicePRData,
                    hostPRData.queue2,hostPRData.queue1.getActiveQueueSize()); 
        }

        hostPRData.queueDlt.resetQueue();
        //allVinG_TraverseVertices<StreamingPageRankOperator::clearVisitedDlt>(custing,devicePRData); //added    
        allVinA_TraverseVertices<StreamingPageRankOperator::clearVisitedDlt>(custing,devicePRData,
                    hostPRData.queueDlt,hostPRData.queueDlt.getActiveQueueSize());
        allVinA_TraverseVertices<StreamingPageRankOperator::clearDelta>(custing,devicePRData,
                    hostPRData.queueDlt,hostPRData.queueDlt.getActiveQueueSize());    
        SyncDeviceWithHost();
        i++;
    }
    #endif
    allVinA_TraverseVertices<StreamingPageRankOperator::prevEqualCurr>(custing,devicePRData,
                        hostPRData.queue2,hostPRData.queue2.getActiveQueueSize());
//---------------------------------------------------------------------------------------------------
}

void StreamingPageRank::UpdateDeletionDiff(cuStinger& custing, BatchUpdateData& bud, length_t* len) {

    length_t batchsize = *(bud.getBatchSize());
    vertexId_t *edgeSrc = bud.getSrc();
    vertexId_t *edgeDst = bud.getDst();
        
    hostPRData.vArraySrc = (vertexId_t*)allocDeviceArray(hostPRData.nv, sizeof(vertexId_t));
    hostPRData.vArrayDst = (vertexId_t*)allocDeviceArray(hostPRData.nv, sizeof(vertexId_t));
    
    copyArrayHostToDevice(edgeSrc,hostPRData.vArraySrc,batchsize,sizeof(vertexId_t));
    copyArrayHostToDevice(edgeDst,hostPRData.vArrayDst,batchsize,sizeof(vertexId_t));


    //insert vertices in updating list in Q1
    for(length_t i=0; i<batchsize; i++) {
        hostPRData.queue1.enqueueFromHost(edgeSrc[i]);
        //cout << "queueu1 enqueue [" << i << "]: " << edgeSrc[i] << endl;
    }    

    copyArrayHostToDevice((length_t*)len,hostPRData.usedOld,hostPRData.nv,sizeof(length_t));

//---------------------------------------------------------------------------------------------------
    printf("\n--------------- update contri \n");
    SyncDeviceWithHost(); //added for threashold and iteration count
    allVinG_TraverseVertices<StreamingPageRankOperator::clear>(custing,devicePRData); //added

//---------------------------------------------------------------------------------------------------  
    printf("--------------- recompute \n");       
    allVinA_TraverseOneEdge<StreamingPageRankOperator::recomputeDeletionContriUndirected>(custing,devicePRData,
                hostPRData.vArraySrc,hostPRData.vArrayDst,hostPRData.queue1.getActiveQueueSize());//batchsize*2);
    SyncHostWithDevice(); //copy Q2 //copyArrayDevicetoHost @@

    allVinA_TraverseVertices<StreamingPageRankOperator::updateDeltaAndMove>(custing,devicePRData,
                hostPRData.queueDlt,hostPRData.queueDlt.getActiveQueueSize());    
    SyncHostWithDevice();
    allVinA_TraverseVertices<StreamingPageRankOperator::updateContributionAndCopy>(custing,devicePRData,
                hostPRData.queue2,hostPRData.queue2.getActiveQueueSize());   
   
    length_t prevEnd = hostPRData.queue2.getQueueEnd(); //1 <-- Q1
    SyncDeviceWithHost();

    hostPRData.queueDlt.resetQueue();
    allVinA_TraverseVertices<StreamingPageRankOperator::clearVisitedDlt>(custing,devicePRData,
                hostPRData.queueDlt,hostPRData.queueDlt.getActiveQueueSize());
    allVinA_TraverseVertices<StreamingPageRankOperator::clearDelta>(custing,devicePRData,
                hostPRData.queueDlt,hostPRData.queueDlt.getActiveQueueSize());    
    SyncDeviceWithHost();

#if 1
    printf("--------------- propagate \n");       
    int i = 0;
    while(hostPRData.queue2,hostPRData.queue2.getActiveQueueSize()>0)
    { 
        printf("++Q2 size (%d): %d\n", i, hostPRData.queue2.getActiveQueueSize());

        allVinA_TraverseEdges_LB<StreamingPageRankOperator::updateContributionsUndirected>(custing,devicePRData,
                    *cusLB,hostPRData.queue2,hostPRData.queue2.getActiveQueueSize());//batchsize
        SyncHostWithDevice(); 
        hostPRData.queue2.setQueueCurr(prevEnd); 
        prevEnd = hostPRData.queue2.getQueueEnd(); //3 <-- 2st step
        SyncDeviceWithHost();
        allVinA_TraverseVertices<StreamingPageRankOperator::updateDeltaAndMove>(custing,devicePRData,
                    hostPRData.queueDlt,hostPRData.queueDlt.getActiveQueueSize());
        allVinA_TraverseVertices<StreamingPageRankOperator::updateContributionAndCopy>(custing,devicePRData,
                    hostPRData.queue2,hostPRData.queue2.getActiveQueueSize());

        if(i == 0){
            allVinA_TraverseVertices<StreamingPageRankOperator::removeContributionsUndirected>(custing,devicePRData,
                    hostPRData.queue2,hostPRData.queue1.getActiveQueueSize()); 
        }

        hostPRData.queueDlt.resetQueue();
        //allVinG_TraverseVertices<StreamingPageRankOperator::clearVisitedDlt>(custing,devicePRData); //added    
        allVinA_TraverseVertices<StreamingPageRankOperator::clearVisitedDlt>(custing,devicePRData,
                    hostPRData.queueDlt,hostPRData.queueDlt.getActiveQueueSize());
        allVinA_TraverseVertices<StreamingPageRankOperator::clearDelta>(custing,devicePRData,
                    hostPRData.queueDlt,hostPRData.queueDlt.getActiveQueueSize());    
        SyncDeviceWithHost();
        i++;
    }
    #endif
    allVinA_TraverseVertices<StreamingPageRankOperator::prevEqualCurr>(custing,devicePRData,
                        hostPRData.queue2,hostPRData.queue2.getActiveQueueSize());
//---------------------------------------------------------------------------------------------------
}
#endif //end of PR_UPDATE


void StreamingPageRank::setInputParameters(length_t prmIterationMax, prType prmthreshold,prType prmDamp){
	hostPRData.iterationMax=prmIterationMax;
	hostPRData.threshold=prmthreshold;
	hostPRData.damp=prmDamp;
	hostPRData.normalizedDamp=(1-hostPRData.damp)/float(hostPRData.nv);
	hostPRData.epsilon= (0.001/float(hostPRData.nv));//0.1% //@@ fix
    printf("##epsilon = %e\n", hostPRData.epsilon);
	SyncDeviceWithHost();
}

length_t StreamingPageRank::getIterationCount(){
	return hostPRData.iteration;
}

#if PR_UPDATE
int fnum = 0;
#endif

void StreamingPageRank::printRankings(cuStinger& custing){
  
	prType* d_scores = (prType*)allocDeviceArray(hostPRData.nv, sizeof(prType));
	vertexId_t* d_ids = (vertexId_t*)allocDeviceArray(hostPRData.nv, sizeof(vertexId_t));

	copyArrayDeviceToDevice(hostPRData.currPR, d_scores,hostPRData.nv, sizeof(prType));

	allVinG_TraverseVertices<StreamingPageRankOperator::setIds>(custing,d_ids);

#if PR_UPDATE
	prType* h_currPr = (prType*)allocHostArray(hostPRData.nv, sizeof(prType));
	copyArrayDeviceToHost(hostPRData.currPR,h_currPr,hostPRData.nv, sizeof(prType));
	
    char nbuff[100];
    sprintf(nbuff, "pr_values_%d.txt", fnum++);
    FILE *fp_npr = fopen(nbuff, "w");
    for (uint64_t v=0; v<hostPRData.nv; v++)
    {
        fprintf(fp_npr,"%d %e\n",v,h_currPr[v]);
    }
   fclose(fp_npr);
#endif //end of PR_UPDATE
	
	standard_context_t context(false);
	mergesort(d_scores,d_ids,hostPRData.nv,greater_t<float>(),context);

	prType* h_scores = (prType*)allocHostArray(hostPRData.nv, sizeof(prType));
	vertexId_t* h_ids    = (vertexId_t*)allocHostArray(hostPRData.nv, sizeof(vertexId_t));

	copyArrayDeviceToHost(d_scores,h_scores,hostPRData.nv, sizeof(prType));
	copyArrayDeviceToHost(d_ids,h_ids,hostPRData.nv, sizeof(vertexId_t));


    for(int v=0; v<10; v++){
            printf("Pr[%d]:= %.10f\n",h_ids[v],h_scores[v]);
    }

#if PR_UPDATE
    char buff[100], buff2[100];
    sprintf(buff, "pr_values_sorted_%d.txt", fnum);
    sprintf(buff2, "pr_values_sorted_no_pr_%d.txt", fnum++);
    FILE *fp_pr = fopen(buff, "w");
    FILE *fp_pr2 = fopen(buff2, "w");
    for (uint64_t v=0; v<hostPRData.nv; v++)
    {
        fprintf(fp_pr,"%d %d %e\n",v,h_ids[v],h_scores[v]);
    	//fprintf(fp_pr2,"%d %d\n",v,h_ids[v]);
    	fprintf(fp_pr2,"%d\n",h_ids[v]);
    }
   fclose(fp_pr);
   fclose(fp_pr2);
#endif //end of PR_UPDATE

//DO NOT REMOVE currPR FOR UPDATE
//	allVinG_TraverseVertices<StreamingPageRankOperator::resetCurr>(custing,devicePRData);
	allVinG_TraverseVertices<StreamingPageRankOperator::sumPr>(custing,devicePRData);

// SyncHostWithDevice();
	prType h_out;

	copyArrayDeviceToHost(hostPRData.reductionOut,&h_out, 1, sizeof(prType));
	cout << "                     h_out: " << setprecision(9) << h_out << endl;
//    cout << "                     " << setprecision(9) << hostPRData.epsilon << endl;

	freeDeviceArray(d_scores);
	freeDeviceArray(d_ids);
	freeHostArray(h_scores);
	freeHostArray(h_ids);
}

}// cuStingerAlgs namespace
