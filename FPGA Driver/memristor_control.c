#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#define DEVICE_FILE "/dev/memristor_array"

int main() {
    int fd;
    char buffer[10];
    u32 row_addr, col_addr, voltage_level;
    u32 current_value;

    // 打开设备文件
    fd = open(DEVICE_FILE, O_RDWR);
    if (fd < 0) {
        perror("Failed to open device file");
        return -1;
    }

    // 设置行地址、列地址和电压等级
    row_addr = 10;
    col_addr = 10;