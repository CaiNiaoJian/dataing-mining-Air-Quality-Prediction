#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
空气质量数据预处理模块

该模块主要完成以下任务：
1. 数据加载和基本信息统计
2. 缺失值检测和处理
3. 时间特征处理
4. 风向特征验证
5. 异常值检测和处理
6. 保存预处理后的数据
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

class DataPreprocessor:
    """数据预处理类"""
    
    def __init__(self, train_path, test_path):
        """
        初始化数据预处理器
        
        Args:
            train_path: 训练数据集路径
            test_path: 测试数据集路径
        """
        self.train_path = train_path
        self.test_path = test_path
        self.train_data = None
        self.test_data = None
        
        # 创建images目录（如果不存在）
        if not os.path.exists('images'):
            os.makedirs('images')
    
    def load_data(self):
        """加载数据并进行基本信息统计"""
        # 加载数据
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)
        
        # 打印基本信息
        print("\n训练集信息：")
        print(self.train_data.info())
        print("\n训练集前5行：")
        print(self.train_data.head())
        print("\n训练集基本统计：")
        print(self.train_data.describe())
        
        # 保存数据基本信息的图表
        self._plot_data_info()
    
    def _plot_data_info(self):
        """绘制数据基本信息的可视化图表"""
        # 数值型特征的分布图
        numeric_features = ['DEWP', 'Ir', 'Is', 'Iws', 'PRES', 'TEMP', 'Label']
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.ravel()
        
        for idx, col in enumerate(numeric_features):
            if col in self.train_data.columns:
                sns.histplot(data=self.train_data, x=col, ax=axes[idx])
                axes[idx].set_title(f'{col}分布图')
        
        plt.tight_layout()
        plt.savefig('images/numeric_features_distribution.png')
        plt.close()
        
        # 风向特征的分布
        wind_features = ['cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv']
        wind_counts = self.train_data[wind_features].sum()
        
        plt.figure(figsize=(10, 6))
        wind_counts.plot(kind='bar')
        plt.title('风向分布')
        plt.xlabel('风向')
        plt.ylabel('计数')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('images/wind_direction_distribution.png')
        plt.close()
    
    def check_missing_values(self):
        """检查缺失值"""
        # 检查训练集缺失值
        train_missing = self.train_data.isnull().sum()
        print("\n训练集缺失值统计：")
        print(train_missing[train_missing > 0])
        
        # 检查测试集缺失值
        test_missing = self.test_data.isnull().sum()
        print("\n测试集缺失值统计：")
        print(test_missing[test_missing > 0])
        
        # 可视化缺失值
        self._plot_missing_values()
        
        # 删除最后一行不完整的数据
        if self.train_data.isnull().any().any():
            self.train_data = self.train_data.dropna()
            print("\n删除缺失值后的训练集形状：", self.train_data.shape)
    
    def _plot_missing_values(self):
        """绘制缺失值可视化图表"""
        plt.figure(figsize=(12, 6))
        sns.heatmap(self.train_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
        plt.title('训练集缺失值分布图')
        plt.tight_layout()
        plt.savefig('images/missing_values_heatmap.png')
        plt.close()
    
    def process_time_features(self):
        """处理时间特征"""
        # 合并date和hour为datetime
        self.train_data['datetime'] = pd.to_datetime(
            self.train_data['date'] + ' ' + self.train_data['hour'].astype(str) + ':00:00'
        )
        self.test_data['datetime'] = pd.to_datetime(
            self.test_data['date'] + ' ' + self.test_data['hour'].astype(str) + ':00:00'
        )
        
        # 提取额外的时间特征
        for df in [self.train_data, self.test_data]:
            df['year'] = df['datetime'].dt.year
            df['month'] = df['datetime'].dt.month
            df['day'] = df['datetime'].dt.day
            df['dayofweek'] = df['datetime'].dt.dayofweek
            df['is_weekend'] = df['datetime'].dt.dayofweek.isin([5, 6]).astype(int)
    
    def verify_wind_direction(self):
        """验证风向特征的独热编码"""
        wind_cols = ['cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv']
        
        # 检查每行风向和是否为1
        train_wind_sum = self.train_data[wind_cols].sum(axis=1)
        test_wind_sum = self.test_data[wind_cols].sum(axis=1)
        
        print("\n训练集风向编码验证：")
        print("每行风向和不为1的记录数：", (train_wind_sum != 1).sum())
        
        print("\n测试集风向编码验证：")
        print("每行风向和不为1的记录数：", (test_wind_sum != 1).sum())
    
    def check_outliers(self):
        """检查异常值"""
        numeric_features = ['DEWP', 'Ir', 'Is', 'Iws', 'PRES', 'TEMP', 'Label']
        
        # 使用箱线图检测异常值
        plt.figure(figsize=(15, 10))
        self.train_data.boxplot(column=numeric_features)
        plt.title('数值特征箱线图')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('images/outliers_boxplot.png')
        plt.close()
        
        # 计算并打印异常值统计
        for feature in numeric_features:
            if feature in self.train_data.columns:
                Q1 = self.train_data[feature].quantile(0.25)
                Q3 = self.train_data[feature].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((self.train_data[feature] < (Q1 - 1.5 * IQR)) | 
                           (self.train_data[feature] > (Q3 + 1.5 * IQR))).sum()
                print(f"\n{feature} 的异常值数量：{outliers}")
    
    def save_processed_data(self, output_path='train.csv'):
        """
        保存预处理后的数据
        
        Args:
            output_path: 输出文件路径，默认为'train.csv'
        """
        # 选择要保存的特征
        features_to_save = [
            'ID', 'datetime', 'year', 'month', 'day', 'hour', 'dayofweek', 'is_weekend',
            'DEWP', 'Ir', 'Is', 'Iws', 'PRES', 'TEMP',
            'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv',
            'Label'
        ]
        
        # 确保所有特征都存在
        features_to_save = [f for f in features_to_save if f in self.train_data.columns]
        
        # 保存处理后的数据
        self.train_data[features_to_save].to_csv(output_path, index=False)
        print(f"\n预处理后的数据已保存至：{output_path}")
        print(f"保存的特征：{', '.join(features_to_save)}")
    
    def process_data(self):
        """执行完整的数据预处理流程"""
        print("开始数据预处理...")
        
        # 1. 加载数据
        print("\n1. 加载数据...")
        self.load_data()
        
        # 2. 检查缺失值
        print("\n2. 检查缺失值...")
        self.check_missing_values()
        
        # 3. 处理时间特征
        print("\n3. 处理时间特征...")
        self.process_time_features()
        
        # 4. 验证风向特征
        print("\n4. 验证风向特征...")
        self.verify_wind_direction()
        
        # 5. 检查异常值
        print("\n5. 检查异常值...")
        self.check_outliers()
        
        # 6. 保存预处理后的数据
        print("\n6. 保存预处理后的数据...")
        self.save_processed_data()
        
        print("\n数据预处理完成！") 