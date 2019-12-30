#define BIN_SIZE 256

typedef struct {
    uchar R;
    uchar G;
    uchar B;
    uchar align;
} RGB;

__kernel
void histogram_kernel(
    __global const RGB* image,
    __local uchar* sharedResult,
    __global uint* globalResultR,
    __global uint* globalResultG,
    __global uint* globalResultB
) {
    size_t localID = get_local_id(0);
    size_t globalID = get_global_id(0);
    size_t groupID = get_group_id(0);
    size_t groupSize = get_local_size(0);

    __local uchar* sharedResultR = sharedResult;
    __local uchar* sharedResultG = sharedResultR + groupSize*BIN_SIZE;
    __local uchar* sharedResultB = sharedResultG + groupSize*BIN_SIZE;

    // Initialization
    uint groupOffset = localID*BIN_SIZE;
    for (int i=0; i<BIN_SIZE; i++) {
        sharedResultR[groupOffset+i] = 0;
        sharedResultG[groupOffset+i] = 0;
        sharedResultB[groupOffset+i] = 0;
    }

    // Calculate the histogram of each work-item
    uint imageOffset = globalID*BIN_SIZE;
    for (int i=0; i<BIN_SIZE; i++) {
        RGB pixel = image[imageOffset+i];
        sharedResultR[groupOffset+pixel.R]++;
        sharedResultG[groupOffset+pixel.G]++;
        sharedResultB[groupOffset+pixel.B]++;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Merge result of each work-item
    uint numberToMergeEach = BIN_SIZE/groupSize;
    for (int i=0; i<numberToMergeEach; i++) {
        uint countR = 0;
        uint countG = 0;
        uint countB = 0;
        for (int j=0; j<groupSize; j++) {
            countR += sharedResultR[j*BIN_SIZE + i*groupSize + localID];
            countG += sharedResultG[j*BIN_SIZE + i*groupSize + localID];
            countB += sharedResultB[j*BIN_SIZE + i*groupSize + localID];
        }
        globalResultR[groupID*BIN_SIZE + i*groupSize + localID] = countR;
        globalResultG[groupID*BIN_SIZE + i*groupSize + localID] = countG;
        globalResultB[groupID*BIN_SIZE + i*groupSize + localID] = countB;
    }
}