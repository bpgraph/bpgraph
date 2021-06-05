#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <test_util.h>

int main() {

    int data[] = {1,2,3,4,5,8};
    size_t length = 6;
    std::cout<<"raw data"<<std::endl;
    for (size_t i = 0 ;i < length ;i++)
    {
        std::cout<<data[i]<<" ";
    }
    std::cout<<std::endl;
    cu_bitmap *bbb = new cu_bitmap(10);
    for (size_t i = 0 ;i < length ;i++)
    {
        bbb->set_bit(data[i]);
    }
    std::cout<<"\nbitmap processed\n";
    for (size_t i = 0 ; i <= 10;i++)
    {
        size_t ret = bbb->get_bit(i);
        std::cout<<i<<":"<<ret << " ";
    }
    std::cout<<"\nbitmap clean\n";
    for (size_t i = 0 ; i <= 10;i++)
    {
        bbb->clean_bit(i);
        size_t ret = bbb->get_bit(i);
        std::cout<<i<<":"<<ret << " ";
    }
    std::cout<<std::endl;
    return 0;

}
