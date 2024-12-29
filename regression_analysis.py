#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回归分析与测试集预测模块

该模块实现以下功能：
1. 加载训练好的模型进行测试集预测
2. 全面的回归分析可视化
3. 模型诊断与评估
4. 预测结果分析与可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from scipy import stats
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置自定义颜色方案
COLOR_PALETTE = ['#6e43b2', '#c9409f', '#fe5880', '#ff8a63', '#ffc256', '#f9f871']
sns.set_palette(COLOR_PALETTE)

class RegressionAnalyzer:
    """回归分析类"""
    
    def __init__(self, train_path='train.csv', test_path='test.csv'):
        """
        初始化回归分析器
        
        Args:
            train_path: 预处理后的训练数据文件路径
            test_path: 预处理后的测试数据文件路径
        """
        self.train_path = train_path
        self.test_path = test_path
        self.train_data = None
        self.test_data = None
        self.model = None
        self.predictions = None
        self.feature_names = None
        
        # 创建可视化结果目录
        if not os.path.exists('images/regression'):
            os.makedirs('images/regression')
    
    def load_data(self):
        """加载预处理后的数据"""
        # 检查文件是否存在
        if not os.path.exists(self.train_path) or not os.path.exists(self.test_path):
            raise FileNotFoundError(
                f"请先运行数据预处理脚本生成 {self.train_path} 和 {self.test_path}"
            )
        
        # 加载训练集和测试集
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)
        
        # 打印数据集信息
        print("\n数据集信息：")
        print(f"训练集形状: {self.train_data.shape}")
        print(f"测试集形状: {self.test_data.shape}")
        
        # 选择特征（确保与训练时使用的特征一致）
        self.feature_names = [
            'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir',
            'year', 'month', 'day', 'hour', 'dayofweek', 'is_weekend',
            'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv'
        ]
        
        # 验证所有特征都存在
        missing_features = [f for f in self.feature_names 
                          if f not in self.train_data.columns 
                          or f not in self.test_data.columns]
        
        if missing_features:
            print(f"\n警告：以下特征在数据集中缺失：{missing_features}")
            # 从特征列表中移除缺失的特征
            self.feature_names = [f for f in self.feature_names 
                                if f not in missing_features]
            print(f"已更新特征列表：{self.feature_names}")
        
        print("\n最终使用的特征：")
        print(self.feature_names)
    
    def load_model(self, model_path='models/RandomForest_model.pkl'):
        """
        加载训练好的模型
        
        Args:
            model_path: 模型文件路径
        """
        self.model = joblib.load(model_path)
        print(f"模型加载完成: {model_path}")
    
    def predict(self):
        """对测试集进行预测"""
        # 准备测试集特征
        X_test = self.test_data[self.feature_names]
        
        # 标准化特征（如果模型需要）
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        
        # 预测
        self.predictions = self.model.predict(X_test_scaled)
        
        # 将预测结果添加到测试集数据中
        self.test_data['Predicted_Label'] = self.predictions
        
        print("\n预测完成")
        print(f"预测结果形状: {self.predictions.shape}")
        print(f"预测值范围: [{self.predictions.min():.2f}, {self.predictions.max():.2f}]")
    
    def plot_feature_relationships(self):
        """绘制特征与目标变量的关系图"""
        n_features = len(self.feature_names)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(20, 5*n_rows))
        
        for i, feature in enumerate(self.feature_names, 1):
            plt.subplot(n_rows, n_cols, i)
            plt.scatter(self.train_data[feature], 
                       self.train_data['Label'],
                       alpha=0.5, 
                       color=COLOR_PALETTE[i % len(COLOR_PALETTE)])
            plt.xlabel(feature)
            plt.ylabel('Label')
            plt.title(f'{feature} vs Label')
        
        plt.tight_layout()
        plt.savefig('images/regression/feature_relationships.png')
        plt.close()
    
    def plot_correlation_matrix(self):
        """绘制相关性矩阵热力图"""
        # 计算相关性矩阵
        corr_matrix = self.train_data[self.feature_names + ['Label']].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.2f')
        plt.title('特征相关性矩阵')
        plt.tight_layout()
        plt.savefig('images/regression/correlation_matrix.png')
        plt.close()
    
    def plot_residuals_analysis(self):
        """残差分析可视化"""
        # 使用训练集进行残差分析
        y_train_pred = self.model.predict(self.train_data[self.feature_names])
        residuals = self.train_data['Label'] - y_train_pred
        
        # 创建一个2x2的子图布局
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # 1. 残差散点图
        axes[0, 0].scatter(y_train_pred, residuals, 
                          alpha=0.5, color=COLOR_PALETTE[0])
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('预测值')
        axes[0, 0].set_ylabel('残差')
        axes[0, 0].set_title('残差散点图')
        
        # 2. 残差直方图
        sns.histplot(residuals, kde=True, ax=axes[0, 1], color=COLOR_PALETTE[1])
        axes[0, 1].set_xlabel('残差')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].set_title('残差分布图')
        
        # 3. Q-Q图
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('残差Q-Q图')
        
        # 4. 标准化残差图
        standardized_residuals = residuals / np.std(residuals)
        axes[1, 1].scatter(y_train_pred, standardized_residuals,
                          alpha=0.5, color=COLOR_PALETTE[3])
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].axhline(y=2, color='r', linestyle=':')
        axes[1, 1].axhline(y=-2, color='r', linestyle=':')
        axes[1, 1].set_xlabel('预测值')
        axes[1, 1].set_ylabel('标准化残差')
        axes[1, 1].set_title('标准化残差图')
        
        plt.tight_layout()
        plt.savefig('images/regression/residuals_analysis.png')
        plt.close()
    
    def plot_prediction_distribution(self):
        """绘制预测值分布"""
        plt.figure(figsize=(12, 6))
        
        # 绘制训练集标签分布
        sns.kdeplot(data=self.train_data['Label'], 
                   label='训练集实际值',
                   color=COLOR_PALETTE[0])
        
        # 绘制测试集预测值分布
        sns.kdeplot(data=self.predictions,
                   label='测试集预测值',
                   color=COLOR_PALETTE[1])
        
        plt.xlabel('值')
        plt.ylabel('密度')
        plt.title('训练集实际值与测试集预测值分布对比')
        plt.legend()
        plt.tight_layout()
        plt.savefig('images/regression/prediction_distribution.png')
        plt.close()
    
    def plot_prediction_time_series(self):
        """绘制时间序列预测结果"""
        # 确保测试集有时间信息
        if 'date' in self.test_data.columns and 'hour' in self.test_data.columns:
            # 创建时间索引
            self.test_data['datetime'] = pd.to_datetime(
                self.test_data['date'] + ' ' + 
                self.test_data['hour'].astype(str) + ':00:00'
            )
            
            plt.figure(figsize=(15, 6))
            plt.plot(self.test_data['datetime'], 
                    self.predictions,
                    label='预测值',
                    color=COLOR_PALETTE[0])
            plt.xlabel('时间')
            plt.ylabel('AQI预测值')
            plt.title('AQI预测时间序列')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('images/regression/prediction_time_series.png')
            plt.close()
    
    def plot_feature_partial_dependence(self):
        """绘制特征部分依赖图"""
        n_features = len(self.feature_names)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(20, 5*n_rows))
        
        for i, feature in enumerate(self.feature_names, 1):
            plt.subplot(n_rows, n_cols, i)
            
            # 计算部分依赖
            feature_values = np.linspace(
                self.train_data[feature].min(),
                self.train_data[feature].max(),
                100
            )
            
            predictions = []
            for value in feature_values:
                X_temp = self.train_data[self.feature_names].copy()
                X_temp[feature] = value
                pred = self.model.predict(X_temp).mean()
                predictions.append(pred)
            
            plt.plot(feature_values, predictions, 
                    color=COLOR_PALETTE[i % len(COLOR_PALETTE)])
            plt.xlabel(feature)
            plt.ylabel('Partial dependence')
            plt.title(f'{feature}的部分依赖图')
        
        plt.tight_layout()
        plt.savefig('images/regression/partial_dependence.png')
        plt.close()
    
    def save_predictions(self):
        """保存预测结果"""
        # 创建完整预测结果数据框
        predictions_df = pd.DataFrame({
            'ID': self.test_data['ID'],
            'Predicted_Label': self.predictions
        })
        
        # 保存完整预测结果
        predictions_df.to_csv('predictions.csv', index=False)
        print("\n完整预测结果已保存至 predictions.csv")
        
        # 创建只包含时间和预测标签的数据框
        label_df = pd.DataFrame({
            'datetime': pd.to_datetime(self.test_data['datetime']),
            'Label': self.predictions
        })
        
        # 按时间排序
        label_df = label_df.sort_values('datetime')
        
        # 保存标签结果
        label_df.to_csv('label.csv', index=False)
        print("时间序列预测结果已保存至 label.csv")
    
    def run_analysis(self):
        """执行完整的回归分析流程"""
        print("开始回归分析...")
        
        # 1. 加载数据
        print("\n1. 加载数据...")
        self.load_data()
        
        # 2. 加载模型
        print("\n2. 加载模型...")
        self.load_model()
        
        # 3. 进行预测
        print("\n3. 进行预测...")
        self.predict()
        
        # 4. 特征关系分析
        print("\n4. 分析特征关系...")
        self.plot_feature_relationships()
        
        # 5. 相关性分析
        print("\n5. 生成相关性矩阵...")
        self.plot_correlation_matrix()
        
        # 6. 残差分析
        print("\n6. 进行残差分析...")
        self.plot_residuals_analysis()
        
        # 7. 预测分布分析
        print("\n7. 分析预测分布...")
        self.plot_prediction_distribution()
        
        # 8. 时间序列分析
        print("\n8. 生成时间序列预测图...")
        self.plot_prediction_time_series()
        
        # 9. 特征部分依赖分析
        print("\n9. 分析���征部分依赖...")
        self.plot_feature_partial_dependence()
        
        # 10. 保存预测结果
        print("\n10. 保存预测结果...")
        self.save_predictions()
        
        print("\n回归分析完成！")

def main():
    """主函数"""
    print("="*50)
    print("开始回归分析与测试集预测...")
    print("="*50)
    
    # 创建回归分析器实例并执行分析
    analyzer = RegressionAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 