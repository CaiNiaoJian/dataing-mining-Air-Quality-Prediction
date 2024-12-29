#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
空气质量数据探索性分析模块

该模块主要完成以下任务：
1. 描述性统计分析
2. 相关性分析
3. 时间序列分析
4. 特征分布分析
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

class ExploratoryAnalysis:
    """数据探索性分析类"""
    
    def __init__(self, data_path='train.csv'):
        """
        初始化数据探索性分析器
        
        Args:
            data_path: 预处理后的数据文件路径
        """
        self.data_path = data_path
        self.data = None
        
        # 创建images目录（如果不存在）
        if not os.path.exists('images'):
            os.makedirs('images')
            
        # 创建images/eda子目录
        if not os.path.exists('images/eda'):
            os.makedirs('images/eda')
    
    def load_data(self):
        """加载数据"""
        self.data = pd.read_csv(self.data_path)
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        print("数据加载完成，形状：", self.data.shape)
    
    def descriptive_statistics(self):
        """描述性统计分析"""
        # 选择数值型特征
        numeric_features = ['DEWP', 'Ir', 'Is', 'Iws', 'PRES', 'TEMP', 'Label']
        
        # 计算描述性统计量
        stats = self.data[numeric_features].describe()
        print("\n描述性统计分析：")
        print(stats)
        
        # 保存结果
        stats.to_csv('images/eda/descriptive_statistics.csv')
        
        # 绘制箱线图
        plt.figure(figsize=(15, 6))
        self.data[numeric_features].boxplot()
        plt.title('数值特征箱线图')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('images/eda/boxplot.png')
        plt.close()
    
    def correlation_analysis(self):
        """相关性分析"""
        # 选择要分析的特征
        features = ['DEWP', 'Ir', 'Is', 'Iws', 'PRES', 'TEMP', 
                   'year', 'month', 'day', 'hour', 'dayofweek', 'is_weekend',
                   'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv', 'Label']
        
        # 计算相关系数
        corr = self.data[features].corr()
        
        # 绘制相关性热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('特征相关性热力图')
        plt.tight_layout()
        plt.savefig('images/eda/correlation_heatmap.png')
        plt.close()
        
        # 打印与Label的相关系数
        print("\n与Label的相关系数：")
        print(corr['Label'].sort_values(ascending=False))
    
    def time_series_analysis(self):
        """时间序列分析"""
        # 按天计算平均AQI
        daily_aqi = self.data.groupby('datetime')['Label'].mean().reset_index()
        
        # 绘制时间序列趋势图
        plt.figure(figsize=(15, 6))
        plt.plot(daily_aqi['datetime'], daily_aqi['Label'])
        plt.title('空气质量指数(AQI)时间序列趋势')
        plt.xlabel('日期')
        plt.ylabel('AQI')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('images/eda/aqi_time_series.png')
        plt.close()
        
        # 按月份分析
        monthly_aqi = self.data.groupby('month')['Label'].agg(['mean', 'std']).reset_index()
        plt.figure(figsize=(10, 6))
        plt.errorbar(monthly_aqi['month'], monthly_aqi['mean'], 
                    yerr=monthly_aqi['std'], fmt='o-')
        plt.title('AQI月度变化')
        plt.xlabel('月份')
        plt.ylabel('平均AQI（误差棒表示标准差）')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('images/eda/aqi_monthly_pattern.png')
        plt.close()
        
        # 按小时分析
        hourly_aqi = self.data.groupby('hour')['Label'].agg(['mean', 'std']).reset_index()
        plt.figure(figsize=(10, 6))
        plt.errorbar(hourly_aqi['hour'], hourly_aqi['mean'],
                    yerr=hourly_aqi['std'], fmt='o-')
        plt.title('AQI日内变化')
        plt.xlabel('小时')
        plt.ylabel('平均AQI（误差棒表示标准差）')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('images/eda/aqi_hourly_pattern.png')
        plt.close()
    
    def feature_distribution_analysis(self):
        """特征分布分析"""
        # 数值型特征的分布
        numeric_features = ['DEWP', 'Ir', 'Is', 'Iws', 'PRES', 'TEMP', 'Label']
        
        # 创建子图
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.ravel()
        
        # 绘制每个特征的分布
        for idx, feature in enumerate(numeric_features):
            if idx < len(axes):
                sns.histplot(data=self.data, x=feature, ax=axes[idx], kde=True)
                axes[idx].set_title(f'{feature}分布')
        
        plt.tight_layout()
        plt.savefig('images/eda/feature_distributions.png')
        plt.close()
        
        # 绘制Label的Q-Q图
        from scipy import stats
        plt.figure(figsize=(8, 8))
        stats.probplot(self.data['Label'], dist="norm", plot=plt)
        plt.title('Label的Q-Q图')
        plt.tight_layout()
        plt.savefig('images/eda/label_qq_plot.png')
        plt.close()
    
    def analyze_data(self):
        """执行完整的探索性分析"""
        print("开始数据探索性分析...")
        
        # 1. 加载数据
        print("\n1. 加载数据...")
        self.load_data()
        
        # 2. 描述性统计分析
        print("\n2. 进行描述性统计分析...")
        self.descriptive_statistics()
        
        # 3. 相关性分析
        print("\n3. 进行相关性分析...")
        self.correlation_analysis()
        
        # 4. 时间序列分析
        print("\n4. 进行时间序列分析...")
        self.time_series_analysis()
        
        # 5. 特征分布分析
        print("\n5. 进行特征分布分析...")
        self.feature_distribution_analysis()
        
        print("\n数据探索性分析完成！")

def main():
    """主函数"""
    print("="*50)
    print("开始数据探索性分析...")
    print("="*50)
    
    # 创建分析器实例并执行分析
    explorer = ExploratoryAnalysis(data_path='train.csv')
    explorer.analyze_data()

if __name__ == "__main__":
    main() 