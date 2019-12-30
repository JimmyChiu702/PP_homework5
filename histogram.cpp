#include <CL/cl.h>

#include <fstream>
#include <iostream>
#include <string>
#include <ios>
#include <vector>

#define BIN_SIZE 256
#define LOCAL_SIZE 16

typedef struct
{
    cl_uchar R;
    cl_uchar G;
    cl_uchar B;
    cl_uchar align;
} RGB;

typedef struct
{
    bool type;
    uint32_t size;
    uint32_t height;
    uint32_t weight;
    RGB *data;
} Image;

Image *readbmp(const char *filename)
{
    std::ifstream bmp(filename, std::ios::binary);
    char header[54];
    bmp.read(header, 54);
    uint32_t size = *(int *)&header[2];
    uint32_t offset = *(int *)&header[10];
    uint32_t w = *(int *)&header[18];
    uint32_t h = *(int *)&header[22];
    uint16_t depth = *(uint16_t *)&header[28];
    if (depth != 24 && depth != 32)
    {
        printf("we don't suppot depth with %d\n", depth);
        exit(0);
    }
    bmp.seekg(offset, bmp.beg);

    Image *ret = new Image();
    ret->type = 1;
    ret->height = h;
    ret->weight = w;
    ret->size = w * h;
    ret->data = new RGB[w * h]{};
    for (int i = 0; i < ret->size; i++)
    {
        bmp.read((char *)&ret->data[i], depth / 8);
    }
    return ret;
}

int writebmp(const char *filename, Image *img)
{
    uint8_t header[54] = {
        0x42,        // identity : B
        0x4d,        // identity : M
        0, 0, 0, 0,  // file size
        0, 0,        // reserved1
        0, 0,        // reserved2
        54, 0, 0, 0, // RGB data offset
        40, 0, 0, 0, // struct BITMAPINFOHEADER size
        0, 0, 0, 0,  // bmp width
        0, 0, 0, 0,  // bmp height
        1, 0,        // planes
        32, 0,       // bit per pixel
        0, 0, 0, 0,  // compression
        0, 0, 0, 0,  // data size
        0, 0, 0, 0,  // h resolution
        0, 0, 0, 0,  // v resolution
        0, 0, 0, 0,  // used colors
        0, 0, 0, 0   // important colors
    };

    // file size
    uint32_t file_size = img->size * 4 + 54;
    header[2] = (unsigned char)(file_size & 0x000000ff);
    header[3] = (file_size >> 8) & 0x000000ff;
    header[4] = (file_size >> 16) & 0x000000ff;
    header[5] = (file_size >> 24) & 0x000000ff;

    // width
    uint32_t width = img->weight;
    header[18] = width & 0x000000ff;
    header[19] = (width >> 8) & 0x000000ff;
    header[20] = (width >> 16) & 0x000000ff;
    header[21] = (width >> 24) & 0x000000ff;

    // height
    uint32_t height = img->height;
    header[22] = height & 0x000000ff;
    header[23] = (height >> 8) & 0x000000ff;
    header[24] = (height >> 16) & 0x000000ff;
    header[25] = (height >> 24) & 0x000000ff;

    std::ofstream fout;
    fout.open(filename, std::ios::binary);
    fout.write((char *)header, 54);
    fout.write((char *)img->data, img->size * 4);
    fout.close();
}

cl_program load_program(cl_context context, const char* filename)
{
    std::ifstream in(filename, std::ios_base::binary);
    if(!in.good()) {
        return 0;
    }

    // get file length
    in.seekg(0, std::ios_base::end);
    size_t length = in.tellg();
    in.seekg(0, std::ios_base::beg);

    // read program source
    std::vector<char> data(length + 1);
    in.read(&data[0], length);
    data[length] = 0;

    // create and build program 
    const char* source = &data[0];
    cl_program program = clCreateProgramWithSource(context, 1, &source, 0, 0);
    if(program == 0) {
        return 0;
    }

    if(clBuildProgram(program, 0, 0, 0, 0, 0) != CL_SUCCESS) {
        return 0;
    }

    return program;
}

void histogram(Image *img, uint32_t R[256], uint32_t G[256], uint32_t B[256]) {
    // Get platform information
    cl_uint num_platforms;
    cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
    cl_platform_id *platforms = new cl_platform_id[num_platforms];
    clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);

    // Get device information
    cl_uint num_devices;
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    cl_device_id *devices = new cl_device_id[num_devices];
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

    // Create context
    cl_context context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &clStatus);

    // Create command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, devices[0], 0, &clStatus);

    // Create buffer
    const uint sharedResultSize = LOCAL_SIZE*BIN_SIZE;
    const uint globalResultSize = (img->size/sharedResultSize + (img->size%sharedResultSize==0 ? 0 : 1))*BIN_SIZE;
    cl_mem clImage = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(RGB)*img->size, NULL, &clStatus);
    clStatus = clEnqueueWriteBuffer(command_queue, clImage, CL_TRUE, 0, sizeof(RGB)*img->size, img->data, 0, NULL, NULL);
    cl_mem globalResultR = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*globalResultSize, NULL, &clStatus);
    cl_mem globalResultG = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*globalResultSize, NULL, &clStatus);
    cl_mem globalResultB = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*globalResultSize, NULL, &clStatus);

    // Create program object
    cl_program program = load_program(context, "histogram.cl");
    cl_kernel histogram_kernel = clCreateKernel(program, "histogram_kernel", &clStatus);

    // Execute OpenCL kernel
    clSetKernelArg(histogram_kernel, 0, sizeof(cl_mem), &clImage);
    clSetKernelArg(histogram_kernel, 1, sizeof(cl_uchar)*sharedResultSize*3, NULL);
    clSetKernelArg(histogram_kernel, 2, sizeof(cl_mem), &globalResultR);
    clSetKernelArg(histogram_kernel, 3, sizeof(cl_mem), &globalResultG);
    clSetKernelArg(histogram_kernel, 4, sizeof(cl_mem), &globalResultB);

    const size_t global_size = (img->size/(BIN_SIZE*LOCAL_SIZE) + (img->size%(BIN_SIZE*LOCAL_SIZE)==0 ? 0 : 1))*LOCAL_SIZE;
    const size_t local_size = LOCAL_SIZE;
    clStatus = clEnqueueNDRangeKernel(command_queue, histogram_kernel, 1, 0, &global_size, &local_size, 0, NULL, NULL);

    uint *resultR = new uint[globalResultSize];
    uint *resultG = new uint[globalResultSize];
    uint *resultB = new uint[globalResultSize];
    clStatus = clEnqueueReadBuffer(command_queue, globalResultR, CL_TRUE, 0, sizeof(uint)*globalResultSize, &resultR[0], 0, NULL, NULL);
    clStatus = clEnqueueReadBuffer(command_queue, globalResultG, CL_TRUE, 0, sizeof(uint)*globalResultSize, &resultG[0], 0, NULL, NULL);
    clStatus = clEnqueueReadBuffer(command_queue, globalResultB, CL_TRUE, 0, sizeof(uint)*globalResultSize, &resultB[0], 0, NULL, NULL);

    clReleaseKernel(histogram_kernel);
    clReleaseProgram(program);
    clReleaseMemObject(clImage);
    clReleaseMemObject(globalResultR);
    clReleaseMemObject(globalResultG);
    clReleaseMemObject(globalResultB);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    std::fill(R, R+256, 0);
    std::fill(G, G+256, 0);
    std::fill(B, B+256, 0);

    for (int i=0; i<globalResultSize/BIN_SIZE; i++) {
        for (int j=0; j<BIN_SIZE; j++) {
            R[j] += resultR[i*BIN_SIZE+j];
            G[j] += resultG[i*BIN_SIZE+j];
            B[j] += resultB[i*BIN_SIZE+j];
        }
    }
}

int main(int argc, char *argv[])
{
    char *filename;
    if (argc >= 2)
    {
        int many_img = argc - 1;
        for (int i = 0; i < many_img; i++)
        {
            filename = argv[i + 1];
            Image *img = readbmp(filename);

            std::cout << img->weight << ":" << img->height << "\n";

            uint32_t R[256];
            uint32_t G[256];
            uint32_t B[256];

            histogram(img,R,G,B);

            int max = 0;
            for(int i=0;i<256;i++){
                max = R[i] > max ? R[i] : max;
                max = G[i] > max ? G[i] : max;
                max = B[i] > max ? B[i] : max;
            }

            Image *ret = new Image();
            ret->type = 1;
            ret->height = 256;
            ret->weight = 256;
            ret->size = 256 * 256;
            ret->data = new RGB[256 * 256];

            for(int i=0;i<ret->height;i++){
                for(int j=0;j<256;j++){
                    if(R[j]*256/max > i)
                        ret->data[256*i+j].R = 255;
                    if(G[j]*256/max > i)
                        ret->data[256*i+j].G = 255;
                    if(B[j]*256/max > i)
                        ret->data[256*i+j].B = 255;
                }
            }

            std::string newfile = "hist_" + std::string(filename); 
            writebmp(newfile.c_str(), ret);
        }
    }else{
        printf("Usage: ./hist <img.bmp> [img2.bmp ...]\n");
    }
    return 0;
}