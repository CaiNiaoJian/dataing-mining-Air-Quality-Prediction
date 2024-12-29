#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
空气质量预测模型构建与评估模块

该模块实现以下功能：
1. 数据准备和特征工程
2. 多种回归模型的实现
3. 模型评估和比较
4. 特征重要性分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置自定义颜色方案
COLOR_PALETTE = ['#6e43b2', '#c9409f', '#fe5880', '#ff8a63', '#ffc256', '#f9f871']
sns.set_palette(COLOR_PALETTE)

class ModelBuilder:
    """模型构建与评估类"""
    
    def __init__(self, data_path='train.csv'):
        """
        初始化模型构建器
        
        Args:
            data_path: 预处理后的数据文件路径
        """
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
        # 创建模型保存目录
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # 创建模型评估结果目录
        if not os.path.exists('images/models'):
            os.makedirs('images/models')
    
    def prepare_data(self):
        """准备建模数据"""
        # 加载数据
        self.data = pd.read_csv(self.data_path)
        
        # 选择特征
        feature_columns = [
            'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir',
            'year', 'month', 'day', 'hour', 'dayofweek', 'is_weekend',
            'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv'
        ]
        
        # 准备特征和目标变量
        self.X = self.data[feature_columns]
        self.y = self.data['Label']
        
        # 数据集划分
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # 特征标准化
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        # 保存特征名称（用于特征重要性分析）
        self.feature_names = feature_columns
        
        print("数据准备完成：")
        print(f"训练集形状: {self.X_train.shape}")
        print(f"测试集形状: {self.X_test.shape}")
    
    def build_models(self):
        """构建多个回归模型"""
        # 线性回归模型
        self.models['Linear'] = LinearRegression()
        
        # Ridge回归（L2正则化）
        self.models['Ridge'] = Ridge(alpha=1.0)
        
        # Lasso回归（L1正则化）
        self.models['Lasso'] = Lasso(alpha=1.0)
        
        # ElasticNet回归（L1+L2正则化）
        self.models['ElasticNet'] = ElasticNet(alpha=1.0, l1_ratio=0.5)
        
        # 随机森林回归
        self.models['RandomForest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # 梯度提升回归
        self.models['GradientBoosting'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        # 支持向量回归
        self.models['SVR'] = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    
    def train_and_evaluate(self):
        """训练和评估模型"""
        for name, model in self.models.items():
            print(f"\n训练模型: {name}")
            
            # 训练模型
            model.fit(self.X_train, self.y_train)
            
            # 预测
            y_pred = model.predict(self.X_test)
            
            # 计算评估指标
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # 交叉验证
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=5, scoring='r2'
            )
            
            # 保存结果
            self.results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            # 保存模型
            joblib.dump(model, f'models/{name}_model.pkl')
            
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R2: {r2:.4f}")
            print(f"CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def analyze_feature_importance(self):
        """分析特征重要性"""
        # 使用随机森林的特征重要性
        rf_model = self.models['RandomForest']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # 绘制特征重要性图
        plt.figure(figsize=(12, 6))
        plt.title('特征重要性排序（随机森林）')
        plt.bar(range(self.X_train.shape[1]), importances[indices])
        plt.xticks(range(self.X_train.shape[1]), [self.feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig('images/models/feature_importance.png')
        plt.close()
        
        # 打印特征重要性
        print("\n特征重要性排序：")
        for f in range(self.X_train.shape[1]):
            print("%d. %s (%f)" % (f + 1, self.feature_names[indices[f]], importances[indices[f]]))
    
    def plot_results(self):
        """绘制模型评估结果"""
        # 绘制预测值与实际值的对比图
        plt.figure(figsize=(15, 10))
        for i, (name, result) in enumerate(self.results.items(), 1):
            plt.subplot(3, 3, i)
            plt.scatter(self.y_test, result['predictions'], alpha=0.5)
            plt.plot([self.y_test.min(), self.y_test.max()],
                    [self.y_test.min(), self.y_test.max()],
                    'r--', lw=2)
            plt.xlabel('实际值')
            plt.ylabel('预测值')
            plt.title(f'{name}\nR2: {result["r2"]:.4f}')
        plt.tight_layout()
        plt.savefig('images/models/prediction_comparison.png')
        plt.close()
        
        # 绘制模型性能对比图
        metrics = ['rmse', 'mae', 'r2']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [result[metric] for result in self.results.values()]
            axes[i].bar(self.results.keys(), values)
            axes[i].set_title(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('images/models/model_comparison.png')
        plt.close()
    
    def build_and_evaluate(self):
        """执行完整的模型构建与评估流程"""
        print("开始模型构建与评估...")
        
        # 1. 准备数据
        print("\n1. 准备数据...")
        self.prepare_data()
        
        # 2. 构建模型
        print("\n2. 构建模型...")
        self.build_models()
        
        # 3. 训练和评估模型
        print("\n3. 训练和评估模型...")
        self.train_and_evaluate()
        
        # 4. 分析特征重要性
        print("\n4. 分析特征重要性...")
        self.analyze_feature_importance()
        
        # 5. 可视化结果
        print("\n5. 生成可视化结果...")
        self.plot_results()
        
        print("\n模型构建与评估完成！")

def main():
    """主函数"""
    print("="*50)
    print("开始模型构建与评估...")
    print("="*50)
    
    # 创建模型构建器实例并执行分析
    builder = ModelBuilder(data_path='train.csv')
    builder.build_and_evaluate()

if __name__ == "__main__":
    main() 