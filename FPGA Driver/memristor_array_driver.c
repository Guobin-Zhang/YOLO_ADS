#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/slab.h>
#include <linux/types.h>
#include <linux/device.h>
#include <linux/io.h>
#include <linux/uaccess.h>

#define DEVICE_NAME "memristor_array"
#define CLASS_NAME "memristor"

static int majorNumber;
static struct class* memristorClass = NULL;
static struct device* memristorDevice = NULL;
static struct cdev memristor_cdev;

// FPGA寄存器地址（假设）
#define FPGA_BASE_ADDR 0x40000000
#define ROW_ADDR_OFFSET 0x00
#define COL_ADDR_OFFSET 0x04
#define VOLTAGE_LEVEL_OFFSET 0x08
#define CURRENT_READ_OFFSET 0x0C
#define CONTROL_OFFSET 0x10

static void __iomem *fpga_base;

// 写入FPGA寄存器
static void write_fpga_register(u32 offset, u32 value) {
    iowrite32(value, fpga_base + offset);
}

// 读取FPGA寄存器
static u32 read_fpga_register(u32 offset) {
    return ioread32(fpga_base + offset);
}

// 打开设备文件
static int memristor_open(struct inode *inode, struct file *file) {
    printk(KERN_INFO "Memristor Array: Device opened\n");
    return 0;
}

// 关闭设备文件
static int memristor_release(struct inode *inode, struct file *file) {
    printk(KERN_INFO "Memristor Array: Device closed\n");
    return 0;
}

// 写操作（用户空间向设备发送命令）
static ssize_t memristor_write(struct file *file, const char __user *buf, size_t count, loff_t *pos) {
    u32 row_addr, col_addr, voltage_level;
    char command[10];

    if (count < 12) {
        printk(KERN_INFO "Memristor Array: Invalid command length\n");
        return -EINVAL;
    }

    if (copy_from_user(command, buf, 10)) {
        printk(KERN_INFO "Memristor Array: Failed to copy command from user space\n");
        return -EFAULT;
    }

    if (sscanf(command, "%d %d %d", &row_addr, &col_addr, &voltage_level) != 3) {
        printk(KERN_INFO "Memristor Array: Invalid command format\n");
        return -EINVAL;
    }

    // 设置行地址
    write_fpga_register(ROW_ADDR_OFFSET, row_addr);
    // 设置列地址
    write_fpga_register(COL_ADDR_OFFSET, col_addr);
    // 设置电压等级
    write_fpga_register(VOLTAGE_LEVEL_OFFSET, voltage_level);

    printk(KERN_INFO "Memristor Array: Write command received: row=%d, col=%d, voltage=%d\n", row_addr, col_addr, voltage_level);
    return count;
}

// 读操作（从设备读取电流数据）
static ssize_t memristor_read(struct file *file, char __user *buf, size_t count, loff_t *pos) {
    u32 current_value;

    // 读取电流值
    current_value = read_fpga_register(CURRENT_READ_OFFSET);

    if (copy_to_user(buf, &current_value, sizeof(u32))) {
        printk(KERN_INFO "Memristor Array: Failed to copy current value to user space\n");
        return -EFAULT;
    }

    printk(KERN_INFO "Memristor Array: Read current value: %d\n", current_value);
    return sizeof(u32);
}

// 文件操作结构体
static struct file_operations fops = {
    .owner = THIS_MODULE,
    .open = memristor_open,
    .release = memristor_release,
    .write = memristor_write,
    .read = memristor_read,
};

// 模块加载函数
static int __init memristor_array_init(void) {
    int result;
    dev_t dev_id;

    // 分配字符设备
    result = alloc_chrdev_region(&dev_id, 0, 1, DEVICE_NAME);
    if (result < 0) {
        printk(KERN_INFO "Memristor Array: Failed to allocate character device region\n");
        return result;
    }

    majorNumber = MAJOR(dev_id);

    // 注册字符设备
    cdev_init(&memristor_cdev, &fops);
    result = cdev_add(&memristor_cdev, dev_id, 1);
    if (result < 0) {
        unregister_chrdev_region(dev_id, 1);
        printk(KERN_INFO "Memristor Array: Failed to add character device\n");
        return result;
    }

    // 创建设备类
    memristorClass = class_create(THIS_MODULE, CLASS_NAME);
    if (IS_ERR(memristorClass)) {
        cdev_del(&memristor_cdev);
        unregister_chrdev_region(dev_id, 1);
        printk(KERN_INFO "Memristor Array: Failed to create device class\n");
        return PTR_ERR(memristorClass);
    }

    // 创建设备
    memristorDevice = device_create(memristorClass, NULL, dev_id, NULL, DEVICE_NAME);
    if (IS_ERR(memristorDevice)) {
        class_destroy(memristorClass);
        cdev_del(&memristor_cdev);
        unregister_chrdev_region(dev_id, 1);
        printk(KERN_INFO "Memristor Array: Failed to create device\n");
        return PTR_ERR(memristorDevice);
    }

    // 映射FPGA寄存器
    fpga_base = ioremap(FPGA_BASE_ADDR, 0x1000);
    if (!fpga_base) {
        device_destroy(memristorClass, dev_id);
        class_destroy(memristorClass);
        cdev_del(&memristor_cdev);
        unregister_chrdev_region(dev_id, 1);
        printk(KERN_INFO "Memristor Array: Failed to map FPGA registers\n");
        return -ENOMEM;
    }

    printk(KERN_INFO "Memristor Array: Driver loaded successfully\n");
    return 0;
}

// 模块卸载函数
static void __exit memristor_array_exit(void) {
    dev_t dev_id = MKDEV(majorNumber, 0);

    // 取消映射FPGA寄存器
    iounmap(fpga_base);

    // 删除设备
    device_destroy(memristorClass, dev_id);
    class_destroy(memristorClass);

    // 删除字符设备
    cdev_del(&memristor_cdev);
    unregister_chrdev_region(dev_id, 1);

    printk(KERN_INFO "Memristor Array: Driver unloaded successfully\n");
}

module_init(memristor_array_init);
module_exit(memristor_array_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("A Linux kernel module for driving a memristor array");
MODULE_VERSION("0.1");