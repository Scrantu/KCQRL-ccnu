import pandas as pd
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt

# --- 定义文件名 ---
DATA_FILE = 'C:\\Users\\zhaoc\\Desktop\\KCQRL-main\\data\\XES3G5M\\question_level\\train_valid_sequences_quelevel.csv'  # 请将此替换为你的数据文件名
OUTPUT_DICT_FILE = 'concept_probabilities.pkl' # 更新：持久化后的字典文件名，现在存储概率

def analyze_and_save_probabilities(data_filepath, output_filepath):
    """
    分析CSV文件中的'concepts'列，统计数字频率，
    然后将频率归一化为概率，并将结果以defaultdict的形式保存到磁盘。
    """
    print(f"开始分析文件: {data_filepath}")

    # --- 步骤 1: 加载数据 ---
    try:
        df = pd.read_csv(data_filepath)
    except FileNotFoundError:
        print(f"警告: 文件 '{data_filepath}' 未找到。将使用内置的示例数据。")
        data = {
            'concepts': [
                "81,336,339_336,81,81,81,78,78,78,81,336,337,78,78,78,78,78",
                "4,195,198,192,192,198,198_192,344,192,198_531,191_192,43,195",
                "0_211,208,208,352,211,208,208,208,272_387,212,212,352,205",
                "0,24_73,9_6,140,10,10,7,66_11_17_326,7,140,140,140,10,366",
                "2,222,301,2,222,223,224,362,361_362_18,247_14,393_18,18,-1"
            ]
        }
        df = pd.DataFrame(data)

    # --- 步骤 2: 频率统计 ---
    concept_frequencies = defaultdict(int)
    for index, row in df.iterrows():
        concepts_str = str(row['concepts'])
        if not concepts_str or concepts_str == 'nan':
            continue
        concept_groups = concepts_str.split(',')
        for group in concept_groups:
            individual_concepts = group.split('_')
            for concept in individual_concepts:
                try:
                    concept_id = int(concept)
                    if concept_id != -1:
                        concept_frequencies[concept_id] += 1
                except ValueError:
                    continue

    print("\n频率统计完成。")

    # --- 步骤 3: 归一化为概率 ---
    total_count = sum(concept_frequencies.values())
    concept_probabilities = defaultdict(float) # 存储概率，所以值是浮点数

    if total_count == 0:
        print("警告: 总频率为零，无法计算概率。")
    else:
        for concept_id, count in concept_frequencies.items():
            concept_probabilities[concept_id] = count / total_count
        print("频率已归一化为概率。")

    # 打印概率最高的前5个概念作为示例
    if concept_probabilities:
        sorted_prob = sorted(concept_probabilities.items(), key=lambda item: item[1], reverse=True)
        print("概率最高的前5个概念:")
        for concept, prob in sorted_prob[:5]:
            print(f"  - 概念ID {concept}: {prob:.6f} (概率)") # 格式化输出，保留6位小数

    # --- 步骤 4: 持久化归一化后的字典到磁盘 ---
    with open(output_filepath, 'wb') as f:
        pickle.dump(concept_probabilities, f)

    print(f"\n成功！概率字典已保存到: {output_filepath}")

    return concept_probabilities


def load_probabilities(filepath):
    """
    从磁盘加载 pickle 文件并返回 defaultdict 对象（现在包含概率）。
    """
    try:
        with open(filepath, 'rb') as f:
            loaded_dict = pickle.load(f)
        print(f"\n已从 '{filepath}' 成功加载字典。")
        print(f"加载的对象类型: {type(loaded_dict)}")
        return loaded_dict
    except FileNotFoundError:
        print(f"错误: 文件 '{filepath}' 不存在。无法加载。")
        return None

def visualize_probabilities(probabilities_dict):
    """
    将概念ID的概率可视化为直方图。
    """
    if not probabilities_dict:
        print("没有概率数据可以可视化。")
        return

    # 提取概念ID和它们的概率
    concept_ids = list(probabilities_dict.keys())
    probabilities = list(probabilities_dict.values())

    plt.figure(figsize=(10, 6)) # 设置图表大小
    plt.hist(concept_ids, bins=50, weights=probabilities, edgecolor='black')
    plt.title('概念ID概率直方图')
    plt.xlabel('概念ID')
    plt.ylabel('概率')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

# --- 主程序执行 ---
if __name__ == "__main__":
    # 1. 运行分析和保存概率功能
    analyzed_probabilities = analyze_and_save_probabilities(DATA_FILE, OUTPUT_DICT_FILE)

    print("\n" + "="*40 + "\n")

    # 2. 运行加载功能来验证
    reloaded_probabilities = load_probabilities(OUTPUT_DICT_FILE)

    if reloaded_probabilities:
        print("\n重新加载的字典内容（前5项）:")
        count = 0
        for concept, prob in reloaded_probabilities.items():
            if count >= 5:
                break
            print(f"  - 概念ID {concept}: {prob:.6f} (概率)")
            count += 1

        # 3. 可视化概率
        visualize_probabilities(reloaded_probabilities)