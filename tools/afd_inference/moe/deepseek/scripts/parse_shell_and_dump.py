import argparse
import yaml

def custom_parse_args():
    parser = argparse.ArgumentParser()
    # 收集所有未知参数
    _, unknown_args = parser.parse_known_args()
    result = {}
    for arg in unknown_args:
        if arg.startswith('--'):
            # 去掉开头的 --
            arg = arg[2:]
            if '=' in arg:
                key, value = arg.split('=', 1)
                key = key.upper()
                try:
                    # 尝试将值转换为整数
                    value = int(value)
                except ValueError:
                    try:
                        # 尝试将值转换为浮点数
                        value = float(value)
                    except ValueError:
                        # 如果转换失败，保持为字符串
                        pass
                result[key] = value

    return result

def main():
    new_file_name = "output.yaml"
    args = custom_parse_args()
    with open("../config/default.yaml", 'r', encoding='utf-8') as file:
        full_data = yaml.safe_load(file)
    # 将解析后的参数保存为 YAML 文件
    full_data.update(args)
    with open(f'../config/{new_file_name}', 'w', encoding="utf-8") as yaml_file:
        yaml.dump(full_data, yaml_file, default_flow_style=False, sort_keys=False)
    print(f"Arguments saved to ../config/{new_file_name}", flush=True)

if __name__ == "__main__":
    main()