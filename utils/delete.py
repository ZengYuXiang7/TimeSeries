# coding : utf-8
# Author : yuxiang Zeng
import os
import glob


def delete_small_log_files(directory):
    # 设置文件路径
    log_files = glob.glob(os.path.join(directory, '*.log'))

    # 遍历所有的.log文件
    for file_path in log_files:
        try:
            # 打开并读取文件
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # 检查行数
            if len(lines) < 5:
                os.remove(file_path)  # 删除文件
                print(f"Deleted '{file_path}' as it had less than 5 lines.")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")


if __name__ == '__main__':
    delete_small_log_files('./')
