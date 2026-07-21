#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <cstdint>
#include <termios.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

int openSerialPort(const std::string& port, speed_t baudRate) {

    int fd = open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd == -1) {
        std::cerr << "无法打开串口: " << port << std::endl;
        return -1;
    }

    // 获取当前串口配置
    struct termios tty;
    if (tcgetattr(fd, &tty) == -1) {
        std::cerr << "无法获取串口属性" << std::endl;
        close(fd);
        return -1;
    }

    // 配置输入输出波特率
    cfsetospeed(&tty, baudRate);
    cfsetispeed(&tty, baudRate);

    // 配置8位数据位，无奇偶校验，1位停止位
    tty.c_cflag &= ~PARENB;  // 禁用奇偶校验
    tty.c_cflag &= ~CSTOPB;  // 1位停止位
    tty.c_cflag &= ~CSIZE;   // 清除数据位设置
    tty.c_cflag |= CS8;      // 8位数据位

    // 禁用硬件流控制
    tty.c_cflag &= ~CRTSCTS;

    // 启用接收器，设置本地模式
    tty.c_cflag |= (CLOCAL | CREAD);

    // 禁用软件流控制
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);

    // 设置原始输入模式
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);

    // 设置原始输出模式
    tty.c_oflag &= ~OPOST;

    // 设置超时
    tty.c_cc[VTIME] = 10;  // 1秒超时
    tty.c_cc[VMIN] = 0;

    // 应用配置
    if (tcsetattr(fd, TCSANOW, &tty) == -1) {
        std::cerr << "无法设置串口属性" << std::endl;
        close(fd);
        return -1;
    }

    return fd;
}

// 读取一行数据
std::string readLine(int fd) {
    std::string line;
    char c;
    ssize_t n;

    while ((n = read(fd, &c, 1)) == 1) {
        if (c == '\n') {
            break;
        }
        line += c;
    }

    return line;
}

int main() {
    // 串口配置
    const std::string port = "/dev/ttyACM0";
    const speed_t baudRate = B19200;

    // 打开串口
    int fd = openSerialPort(port, baudRate);
    if (fd == -1) {
        return 1;
    }

    std::cout << "开始接收OpenMV数据..." << std::endl;

    try {
        while (true) {
            // 读取一行数据
            std::string line = readLine(fd);

            // 读取单个字节
            uint8_t byte_data;
            ssize_t n = read(fd, &byte_data, 1);

            // 处理单个字节
            if (n > 0) {
                std::cout << "收到字节: 0x" << std::hex << static_cast<int>(byte_data) << std::dec << std::endl;

                // 转换为二进制字符串
                std::string bit_string;
                for (int i = 7; i >= 0; --i) {
                    bit_string += ((byte_data >> i) & 1) ? '1' : '0';
                }
                std::cout << "按位解码: " << bit_string << std::endl;
                std::cout << "---" << std::endl;
            }

            // 处理一行数据
            if (!line.empty()) {
                std::cout << "收到信息: " << line << std::endl;
            }

            // 延迟100ms
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    } catch (const std::exception& e) {
        std::cerr << "发生错误: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "程序已停止" << std::endl;
    }

    // 关闭串口
    close(fd);
    std::cout << "串口已关闭" << std::endl;

    return 0;
}
