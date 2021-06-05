#ifndef DEVICE_CUH_
#define DEVICE_CUH_

#include "cuda_error_check.cuh"
#include "buffers.cuh"
#include "globals.h"


template <typename idxT, typename vT, class eT, class vStaticT>
class cuda_device{

public:
	// Variables to mark boundary vertices.
	std::vector< idxT > vertexIndicesToMoveVec;	// Used only in MS.
	uint nVerticesToSend, nVerticesToReceive;
	unsigned long long globalMemCapacity;

	// Device-specific CUDA stream.
	cudaStream_t devStream;

	// Necessary device and host CSR buffers.
	device_buffer< vT > vertexValue;
	device_buffer< idxT > edgesIndices;
	device_buffer< idxT > vertexIndices;
	device_buffer< eT > EdgeValue;
	device_buffer< vStaticT > VertexValueStatic;
	host_pinned_buffer<int> finishedHost;
	device_buffer<int> finished;


	// Outbox buffers. They will be used in multi-GPU scenarios for MS and VR.
	uint vertexOffset, edgeOffset, nDedicatedVertices, nDedicatedEdges;
	device_buffer<uint> outboxTop;
	device_buffer< idxT > outboxIndices;
	device_buffer< vT > outboxVertices;

	device_buffer< idxT > inboxIndices_odd;
	device_buffer< vT > inboxVertices_odd;
	host_pinned_buffer<uint> inboxTop_odd;
	device_buffer< idxT > inboxIndices_even;
	device_buffer< vT > inboxVertices_even;
	host_pinned_buffer<uint> inboxTop_even;

	// MS Temporary host buffer.
	host_pinned_buffer< idxT > tmpHostIndices;


	cuda_device():
		vertexIndicesToMoveVec( 0 ), nVerticesToSend( 0 ), nVerticesToReceive( 0 ),
		vertexOffset( 0 ), edgeOffset( 0 ), nDedicatedVertices( 0 ), nDedicatedEdges( 0 )
	{}
	void create_device_stream(){
		CUDAErrorCheck( cudaStreamCreate( &devStream ) );
	}
	void destroy_device_stream(){
		CUDAErrorCheck( cudaStreamDestroy( devStream ) );
	}

	// Essential CSR buffers.
	void allocate_CSR_buffers( const uint nVertices ) {

		vertexValue.alloc( nVertices );	// A full version of vertices for every device.
		edgesIndices.alloc( nDedicatedVertices + 1 );
		vertexIndices.alloc( nDedicatedEdges );
		if( sizeof(eT) > 1 ) EdgeValue.alloc( nDedicatedEdges );
		if( sizeof(vStaticT) > 1 ) VertexValueStatic.alloc( nVertices );
		finished.alloc( 1 );
		finishedHost.alloc( 1 );

	}

};


#endif