#ifndef GET_DEVICES_CUH
#define GET_DEVICES_CUH

void getIdenticalGPUs(int num_of_gpus, std::set<int> &identicalGPUs) {
  int *major_minor = (int *)malloc(sizeof(int) * num_of_gpus * 2);
  int foundIdenticalGPUs = 0;

  for (int i = 0; i < num_of_gpus; i++) {
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, i));
    major_minor[i * 2] = deviceProp.major;
    major_minor[i * 2 + 1] = deviceProp.minor;
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n", i,
           deviceProp.name, deviceProp.major, deviceProp.minor);
  }

  int maxMajorMinor[2] = {0, 0};

  for (int i = 0; i < num_of_gpus; i++) {
    for (int j = i + 1; j < num_of_gpus; j++) {
      if ((major_minor[i * 2] == major_minor[j * 2]) &&
          (major_minor[i * 2 + 1] == major_minor[j * 2 + 1])) {
        identicalGPUs.insert(i);
        identicalGPUs.insert(j);
        foundIdenticalGPUs = 1;
        if (maxMajorMinor[0] < major_minor[i * 2] &&
            maxMajorMinor[1] < major_minor[i * 2 + 1]) {
          maxMajorMinor[0] = major_minor[i * 2];
          maxMajorMinor[1] = major_minor[i * 2 + 1];
        }
      }
    }
  }

  free(major_minor);
  if (!foundIdenticalGPUs) {
    printf(
        "No Two or more GPUs with same architecture found\nWaiving the "
        "sample\n");
    exit(EXIT_WAIVED);
  }

  std::set<int>::iterator it = identicalGPUs.begin();

  // Iterate over all the identical GPUs found
  while (it != identicalGPUs.end()) {
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, *it));
    // Remove all the GPUs which are less than the best arch available
    if (deviceProp.major != maxMajorMinor[0] &&
        deviceProp.minor != maxMajorMinor[1]) {
      identicalGPUs.erase(it);
    }
    if (!deviceProp.cooperativeMultiDeviceLaunch ||
        !deviceProp.concurrentManagedAccess) {
      identicalGPUs.erase(it);
    }
    it++;
  }

  return;
}


#endif