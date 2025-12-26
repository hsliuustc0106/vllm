import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="llm run parameters")
    parser.add_argument('--yaml_file', type=str, default="default.yaml", help="name of yaml file")
    parser_args = parser.parse_args()
    return parser_args

def read_yaml_and_export(yaml_file_path):
    try:
        with open(yaml_file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            if data is not None:
                for key, value in data.items():
                    # 输出 export 命令
                    print(f'export {key}="{value}"')
            else:
                print("YAML 文件为空。")
    except FileNotFoundError:
        print(f"未找到指定的 YAML 文件: {yaml_file_path}")
    except yaml.YAMLError as e:
        print(f"解析 YAML 文件时出错: {e}")


if __name__ == "__main__":
    args = parse_args()
    read_yaml_and_export(f"../config/{args.yaml_file}")