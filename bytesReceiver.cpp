#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <thread>
#include <mutex>
#include <queue>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>

using namespace std;
bool isFileExists_stat(std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}
int main(){
    string filename = "/Users/harrywang/Documents/pipe_trans/pipe_dir/pipe2";
    if(!isFileExists_stat(filename)){
        std::cout << filename << std::endl;
        int32_t ret = mkfifo(filename.c_str(), S_IFIFO | 0666);
        if (ret == -1){
            std::cout << errno << std::endl;
            std::cout << "Make fifo error\n";
            return -1;
        }
    }
    
    int fd = open(filename.c_str(), O_RDONLY);
    if(fd < 0){
        return fd;
    }
    const int32_t bufSize = 8192;
    char readBuf[bufSize];
    int i = 0;
    long total_time = 0;
    struct timeval tv;
    while(1){
        gettimeofday(&tv, NULL);
        long now1 = tv.tv_sec * 1000 + tv.tv_usec / 1000; /* get milliseconds */
        int ret = read(fd, readBuf, bufSize);
        gettimeofday(&tv, NULL);
        long now2 = tv.tv_sec * 1000 + tv.tv_usec / 1000; /* get milliseconds */
        if(ret > 0){
            total_time += now2 - now1;
            if(i % 50000 == 1)
                cout << now2 - now1 << "ms " << total_time << "ms" << endl;
        }
        i++;
        
    }

    
    return 0;


}
