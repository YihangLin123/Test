import pandas as pd


def clean_excel_data(file_path, output_path):
    # 1. 读取 Excel 文件（默认读取第一个工作表）
    df = pd.read_excel(file_path)

    # 2. 定义第E列至第AH列的索引范围
    # A=0, B=1, C=2, D=3, E=4
    # AH=33（在Python切片中结束索引取34，即包含第AH列）
    start_idx = 4
    end_idx = 34 + 1

    # 获取需要检查空值的列名列表
    target_columns = df.columns[start_idx:end_idx]

    print(f"原始数据总行数: {len(df)}")
    print(f"正在检查的指标列范围: {target_columns[0]} 到 {target_columns[-1]} (共 {len(target_columns)} 列)")

    # 3. 删除指定列中有空白值（NaN）的样本行
    df_cleaned = df.dropna(subset=target_columns)

    print(f"清理后的数据总行数: {len(df_cleaned)}")
    print(f"已删除包含空白值的样本数: {len(df) - len(df_cleaned)}")

    # 4. 将清理后的数据保存到新文件
    df_cleaned.to_excel(output_path, index=False)
    print(f"处理后的文件已成功保存至: {output_path}")


# 执行函数
if __name__ == "__main__":
    input_file = "./data/original_data_cleaned_C13_C14.xlsx"
    output_file = "./data/original_data_cleaned_C13_C14_processed.xlsx"
    clean_excel_data(input_file, output_file)