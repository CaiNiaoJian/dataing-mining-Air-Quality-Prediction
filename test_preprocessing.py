#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试集数据预处理模块

该模块实现以下功能：
1. 测试集数据加载和基本信息统计
2. 缺失值处理
3. 时间特征处理
4. 风向特征验证
5. 异常值检测
6. 保存预处理后的测试集
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置自定义颜色方案
COLOR_PALETTE = ['#6e43b2', '#c9409f', '#fe5880', '#ff8a63', '#ffc256', '#f9f871']
sns.set_palette(COLOR_PALETTE)

class TestDataPreprocessor:
    """测试集数据预处理类"""
    
    def __init__(self, test_path='test_noLabelOX.csv'):
        """
        初始化数据预处理器
        
        Args:
            test_path: 测试集文件路径
        """
        self.test_path = test_path
        self.test_data = None
        
        # 创建预处理可视化结果目录
        if not os.path.exists('images/test_preprocessing'):
            os.makedirs('images/test_preprocessing')
    
    def load_data(self):
        """加载数据并显示基本信息"""
        # 加载测试集
        self.test_data = pd.read_csv(self.test_path)
        
        print("\n数据基本信息：")
        print(f"测试集形状: {self.test_data.shape}")
        print("\n列名：")
        print(self.test_data.columns.tolist())
        print("\n数据类型：")
        print(self.test_data.dtypes)
    
    def handle_missing_values(self):
        """处理缺失值"""
        # 检查缺失值
        missing_stats = self.test_data.isnull().sum()
        
        print("\n缺失值统计：")
        print(missing_stats[missing_stats > 0])
        
        # 删除最后一行（如果不完整）
        if self.test_data.iloc[-1].isnull().any():
            self.test_data = self.test_data.iloc[:-1].copy()
            print("已删除最后一行不完整数据")
        
        # 绘制缺失值热力图
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.test_data.isnull(), 
                   yticklabels=False,
                   cmap='viridis',
                   cbar_kws={'label': '缺失值'})
        plt.title('缺失值分布热力图')
        plt.tight_layout()
        plt.savefig('images/test_preprocessing/missing_values_heatmap.png')
        plt.close()
    
    def process_time_features(self):
        """处理时间特征"""
        # 转换日期为datetime类型
        self.test_data['datetime'] = pd.to_datetime(self.test_data['date'])
        
        # 提取时间特征
        self.test_data['year'] = self.test_data['datetime'].dt.year
        self.test_data['month'] = self.test_data['datetime'].dt.month
        self.test_data['day'] = self.test_data['datetime'].dt.day
        self.test_data['dayofweek'] = self.test_data['datetime'].dt.dayofweek
        self.test_data['is_weekend'] = self.test_data['dayofweek'].isin([5, 6]).astype(int)
        
        print("\n时间特征处理完成")
    
    def verify_wind_direction(self):
        """验证风向特征"""
        # 检查风向编码是否正确
        wind_columns = ['cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv']
        wind_sum = self.test_data[wind_columns].sum(axis=1)
        
        if not all(wind_sum == 1):
            print("\n警告：存在风向编码异常！")
            print(f"异常记录数: {sum(wind_sum != 1)}")
        else:
            print("\n风向特征验证通过")
        
        # 绘制风向分布图
        plt.figure(figsize=(10, 6))
        wind_dist = self.test_data[wind_columns].sum()
        plt.bar(wind_columns, wind_dist, color=COLOR_PALETTE)
        plt.title('风向分布')
        plt.ylabel('频数')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('images/test_preprocessing/wind_direction_distribution.png')
        plt.close()
    
    def detect_outliers(self):
        """检测异常值"""
        # 数值特征
        numeric_features = ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
        
        # 绘制箱线图
        plt.figure(figsize=(12, 6))
        self.test_data[numeric_features].boxplot()
        plt.title('数值特征箱线图')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('images/test_preprocessing/outliers_boxplot.png')
        plt.close()
        
        # 计算异常值数量
        outliers_stats = {}
        for feature in numeric_features:
            Q1 = self.test_data[feature].quantile(0.25)
            Q3 = self.test_data[feature].quantile(0.75)
            IQR = Q3 - Q1
            outliers = sum((self.test_data[feature] < (Q1 - 1.5 * IQR)) | 
                          (self.test_data[feature] > (Q3 + 1.5 * IQR)))
            outliers_stats[feature] = outliers
        
        print("\n异常值统计：")
        for feature, count in outliers_stats.items():
            print(f"{feature}: {count}个异常值")
    
    def save_processed_data(self, output_path='test.csv'):
        """
        保存预处理后的数据
        
        Args:
            output_path: 输出文件路径
        """
        # 选择要保存的特征
        features_to_save = [
            'ID', 'datetime', 'year', 'month', 'day', 'hour', 'dayofweek', 'is_weekend',
            'DEWP', 'Ir', 'Is', 'Iws', 'PRES', 'TEMP',
            'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv'
        ]
        
        # 确保所有特征都存在
        features_to_save = [f for f in features_to_save if f in self.test_data.columns]
        
        # 保存处理后的数据
        self.test_data[features_to_save].to_csv(output_path, index=False)
        print(f"\n预处理后的数据已保存至：{output_path}")
        print(f"保存的特征：{', '.join(features_to_save)}")
    
    def process_data(self):
        """执行完整的数据预处理流程"""
        print("开始测试集数据预处理...")
        
        # 1. 加载数据
        print("\n1. 加载数据...")
        self.load_data()
        
        # 2. 处理缺失值
        print("\n2. 处理缺失值...")
        self.handle_missing_values()
        
        # 3. 处理时间特征
        print("\n3. 处理时间特征...")
        self.process_time_features()
        
        # 4. 验证风向特征
        print("\n4. 验证风向特征...")
        self.verify_wind_direction()
        
        # 5. 检测异常值
        print("\n5. 检测异常值...")
        self.detect_outliers()
        
        # 6. 保存预处理后的数据
        print("\n6. 保存预处理后的数据...")
        self.save_processed_data()
        
        print("\n测试集数据预处理完成！")

def main():
    """主函数"""
    print("="*50)
    print("开始测试集数据预处理...")
    print("="*50)
    
    # 创建预处理器实例并执行预处理
    preprocessor = TestDataPreprocessor()
    preprocessor.process_data()

if __name__ == "__main__":
    main() 