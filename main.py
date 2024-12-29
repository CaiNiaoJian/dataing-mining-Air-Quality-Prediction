#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
空气质量数据分析主程序

该程序用于执行数据预处理和分析流程
"""

from data_preprocessing import DataPreprocessor
from exploratory_analysis import ExploratoryAnalysis

def main():
    """主函数"""
    # 1. 数据预处理
    print("="*50)
    print("开始数据预处理...")
    print("="*50)
    
    preprocessor = DataPreprocessor(
        train_path='trainOX.csv',
        test_path='test_noLabelOX.csv'
    )
    preprocessor.process_data()
    
    # 2. 数据探索性分析
    print("\n"+"="*50)
    print("开始数据探索性分析...")
    print("="*50)
    
    explorer = ExploratoryAnalysis(data_path='train.csv')
    explorer.analyze_data()

if __name__ == "__main__":
    main() 