import sys
import json
if __name__ == '__main__':
    # 输入
    # >> cat qq.json | python prepare_file.py > qq1.json
    # 读取qq.json做为输入 经过 prepare_file 中的操作，所有的print输出做为qq1.json中的文件
    i = 0
    for line in sys.stdin:
        i = i + 1
        if i > 2 * 1:
            break
        if i < 2 * 0:
            continue
        sample = json.loads(line)
        print(json.dumps(sample, ensure_ascii=False))