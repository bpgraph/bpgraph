/*************************************************************************
    > File Name: memory_monitor.h
    > Author: ZhangHeng
    > Mail: heng200888@163.com 
*/

#ifndef MEMORY_MONITOR_H_
#define MEMORY_MONITOR_H_

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <thread>

#include <unistd.h>
#include <sys/resource.h>
#include <stdio.h>

// Gets snapshot of the memory used and check if it is higher than the recorded peak
class memory_monitor {
private:
	size_t peakMemory;
	bool enabled;

public:
	memory_monitor()
	{
		peakMemory = 0;
		enabled = true;
	}

	~memory_monitor(){}

	void update_peak_memory()
	{
		std::chrono::milliseconds delay(200);
		size_t currentMem;

		while (enabled) {
			currentMem = get_current_memory();
			if (currentMem > peakMemory) {
				peakMemory = currentMem;
			}
			std::this_thread::sleep_for(delay);
		}
		return;
	}

	
	size_t get_peak_memory()
	{
		if (peakMemory == 0) {
			return get_current_memory();
		}
		return peakMemory;
	}


	size_t get_current_memory()
	{
		long rss = 0L;
		FILE* fp = NULL;
		if ((fp = fopen("/proc/self/statm", "r")) == NULL)
			return (size_t)0L;      /* Fail to open? */
		if (fscanf(fp, "%*s%ld", &rss) != 1)
		{
			fclose(fp);
			return (size_t)0L;      /* Fail to read? */
		}
		fclose(fp);
		return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);
	}

	void start_monitoring()
	{
		peakMemory = 0;
	}

	void stop_monitoring()
	{
		enabled = false;
	}



};

#endif
