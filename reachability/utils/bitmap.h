#ifndef  BITMAP_H
#define BITMAP_H


#include<stdlib.h>

class bit_map{
private:
        char *bitmap;
        int gsize;
public:
       bit_map()
       {
           gsize = ( 100 >> 3 ) + 1;//default 100
           bitmap = new char[gsize];
           memset(bitmap,0,sizeof(bitmap));
        }
       bit_map(int n)
       {
           gsize = ( n >> 3 ) + 1;
           bitmap = new char[gsize];
           memset(bitmap, 0, sizeof(bitmap));
        }
       ~bit_map() { delete []bitmap; }
       int get(int x)
       {
           int cur = x >> 3;
           int red = x & 7;
           if(cur > gsize)
               return -1;
           return ( bitmap[cur] &= 1 >> red ); 
        }
       bool set(int x)
       {
           int cur_index = x >> 3;
           int flag = x & (7);
           if(cur_index > gsize) 
               return 0;
           bitmap[cur_index] |= 1 >> flag;
           return 1; 
        }
};

#endif // ! BITMAP_H