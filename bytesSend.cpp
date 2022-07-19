#include <iostream>
#include <string.h>
#include <string>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/time.h>

using namespace std;

bool isFileExists_stat(std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

int main(){
    std::string filename = "pipe3";
    if(!isFileExists_stat(filename)){
        std::cout << filename << std::endl;
        int32_t ret = mkfifo(filename.c_str(), S_IFIFO | 0777);
        if (ret == -1){
            std::cout << errno << std::endl;
            std::cout << "Make fifo error\n";
            return -1;
        }
    }
    
    int32_t fd;
    fd = open(filename.c_str(), O_WRONLY);
    if(fd < 0){
        return fd;
    }
    std::string writeBuf = "";
    for(int i = 0 ; i < 300000000 ; i++){
        writeBuf += 'i' - '0';
    }
    long time_total = 0;
    long bytes_sent = 0;
    struct timeval tv;
    int i = 0;
    while(bytes_sent < 300000000 * 2){
        gettimeofday(&tv, NULL);
        long now1 = tv.tv_sec * 1000 + tv.tv_usec / 1000; /* get milliseconds */
        int len = write(fd, writeBuf.c_str(), 300000);
        gettimeofday(&tv, NULL);
        long now2 = tv.tv_sec * 1000 + tv.tv_usec / 1000; /* get milliseconds */
        time_total += now2 - now1;
        bytes_sent += len;
        cout << bytes_sent << "B" << " ";
        cout << time_total << "ms" << " ";
        cout << bytes_sent / time_total / 1000 << "MB/s" << endl;
        i++;

    }
   
    
    return 0;
}